from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_stage1_r15_g import adapter_off_model, merged_linear_model, projection_diagnostics
from mmdd_stage1.models import StudentJoinabilityModel, add_projection_residual
from recover_stage1_r15_training_source import reconstruct, reconstruct_r14, recover
from run_stage1_r15 import optimizer_audit


def test_linear_residual_merge_preserves_all_five_relation_scores():
    torch.manual_seed(13)
    base = StudentJoinabilityModel(8, 4, initialization="random")
    residual = add_projection_residual(
        base,
        "linear",
        hidden_dim=3,
        scales={"table": 0.7, "text": 1.3, "image": 2.1},
    )
    with torch.no_grad():
        for layer in residual.projection_residual_outputs.values():
            layer.weight.normal_(std=0.2)
    merged = merged_linear_model(residual)
    source = torch.randn(7, 8)
    destination = torch.randn(7, 8)
    for source_type, destination_type in (
        ("table", "table"),
        ("table", "text"),
        ("table", "image"),
        ("text", "table"),
        ("image", "table"),
    ):
        torch.testing.assert_close(
            residual.score_embeddings(
                source, source_type, destination, destination_type
            ),
            merged.score_embeddings(
                source, source_type, destination, destination_type
            ),
            atol=1e-5,
            rtol=1e-5,
        )


def test_adapter_off_keeps_trained_base_and_relations_only():
    base = StudentJoinabilityModel(4, 4, initialization="identity")
    residual = add_projection_residual(base, "gelu", hidden_dim=2)
    with torch.no_grad():
        residual.projection_residual_outputs["table"].weight.fill_(0.5)
    disabled = adapter_off_model(residual)
    embedding = torch.tensor([1.0, -2.0, 0.5, 3.0])
    torch.testing.assert_close(
        disabled.project(embedding, "table"),
        residual.projections["table"](embedding),
    )
    assert not torch.equal(
        disabled.project(embedding, "table"), residual.project(embedding, "table")
    )


def test_optimizer_audit_exports_parameter_membership_and_state_steps():
    model = StudentJoinabilityModel(4, 4, initialization="identity")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.relation_parameters(), "lr": 1e-5},
            {"params": model.projection_parameters(), "lr": 1e-6},
        ],
        weight_decay=0.01,
    )
    before = optimizer_audit(optimizer, model)
    assert before["groups"][0]["state_entries"] == 0
    assert before["groups"][1]["parameter_names"] == [
        "projections.table.weight",
        "projections.text.weight",
        "projections.image.weight",
    ]

    model.project(torch.ones(4), "table").sum().backward()
    optimizer.step()
    after = optimizer_audit(optimizer, model)
    assert after["groups"][1]["state_entries"] == 1
    assert after["groups"][1]["state_step_min"] == 1
    assert after["groups"][1]["state_step_max"] == 1
    assert after["gradient_clipping"] is None


def test_linear_merge_rejects_nonlinear_adapter():
    model = add_projection_residual(
        StudentJoinabilityModel(4, 4, initialization="identity"),
        "gelu",
        hidden_dim=2,
    )
    with pytest.raises(ValueError, match="linear residual"):
        merged_linear_model(model)


def test_adapter_off_geometry_reports_actual_disabled_output():
    from types import SimpleNamespace

    model = add_projection_residual(
        StudentJoinabilityModel(4, 4, initialization="identity"),
        "gelu", hidden_dim=2,
    )
    with torch.no_grad():
        model.projection_residual_outputs["table"].weight.fill_(0.5)
    embeddings = torch.eye(4)
    store = SimpleNamespace(
        embedding_features=lambda object_id: SimpleNamespace(
            embedding=embeddings[int(object_id)]
        )
    )
    disabled = adapter_off_model(model)
    geometry = projection_diagnostics(
        disabled, model, disabled, store, {"table": ["0", "1", "2", "3"]},
        torch.device("cpu"), sample_size=4,
    )["by_type"]["table"]
    assert geometry["residual_enabled"] is False
    assert geometry["residual_output"]["rms"] == 0
    assert geometry["full_output"]["rms"] == geometry["base_output"]["rms"]
    assert len(geometry["base_output"]["mean_vector"]) == 4
    assert len(geometry["full_output"]["mean_vector"]) == 4
    assert len(geometry["s0_output"]["mean_vector"]) == 4
    assert geometry["full_output"]["mean_vector"] == geometry["base_output"]["mean_vector"]


@pytest.mark.parametrize("name,reconstruct_source,expected_hash", [
    ("run_stage1_r15.py", reconstruct,
     "a1a8b4f3fc3796a00d086928b2799644b3160f722b2e695c27a90e6a988811cc"),
    ("run_stage1_r14.py", reconstruct_r14,
     "da774e9fb48c8bdbcb4447f5063722f955e2a67582c2e764879868141f499a67"),
])
def test_reconstructed_entrypoint_matches_frozen_training_hash(name, reconstruct_source, expected_hash):
    current = (ROOT / "src" / name).read_text(encoding="utf-8")
    original = reconstruct_source(current)
    assert hashlib.sha256(original.encode()).hexdigest() == expected_hash
    functions = [{node.name: ast.dump(node, include_attributes=False)
                  for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
                 for text in (current, original)]
    assert "evaluate_arm" not in functions[1]
    for name in set(functions[1]) - {"main", "parse_args"}:
        assert functions[0][name] == functions[1][name]


def test_source_recovery_rejects_manifest_hash_mismatch_before_writing(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/run_stage1_r15.py").write_text(
        (ROOT / "src/run_stage1_r15.py").read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "work/stage1_optimization_r15_20260909"
    for arm in ("l_eoff", "n_eoff"):
        directory = output / f"stageI_interaction/{arm}_seed13"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"code_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="both training manifest hashes"):
        recover(tmp_path)
    assert not (output / "source_snapshot").exists()

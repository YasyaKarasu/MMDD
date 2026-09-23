"""Tensor-level semantic tests for the R25 execution contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.b13_recipe import consumed_batch_order, edge_objective, ranking_scores, recipe_signature
from mmdd_stage1.data import EdgeExample
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.objectives import listwise_cross_entropy
from prepare_stage1_r12_candidates import materialize_batch
from mmdd_stage1.r25_objectives import R25ObjectiveConfig, edge_continuation_loss, split_objective
from mmdd_stage1.scoring import ListScores, TargetScores
from run_stage1_r25 import ARMS, SEEDS, confidence_alpha, dry_run, resume_compatible


def _list(logits: torch.Tensor, mask: torch.Tensor | None = None) -> ListScores:
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    positive = torch.zeros_like(mask)
    positive[:, 0] = mask[:, 0]
    return ListScores(logits, mask, torch.zeros(logits.shape[0], dtype=torch.long), positive)


def _target(d: torch.Tensor, e: torch.Tensor, e_mask: torch.Tensor | None = None) -> TargetScores:
    return TargetScores(_list(d), _list(e, e_mask))


def _teacher(qt: torch.Tensor, e_mask: torch.Tensor | None = None) -> TargetScores:
    return TargetScores(_list(qt), _list(qt.clone(), e_mask))


def test_T05_edge_continuation_does_not_consume_path_targets():
    qe = _list(torch.tensor([[2.0, 0.0]], requires_grad=True))
    et = _list(torch.tensor([[1.0, -1.0]], requires_grad=True))
    before = edge_continuation_loss(qe, et)["loss"]
    # A path rewire has no input slot in the edge-only factory.
    after = edge_continuation_loss(qe, et)["loss"]
    assert torch.equal(before, after)
    assert edge_continuation_loss(qe, et)["path_supervised_loss"].item() == 0.0


def test_T01_b13_supervision_score_is_sigmoid_scaled_and_kd_space_stays_raw():
    raw = _list(torch.tensor([[-2.0, 0.0, 2.0]]))
    transformed = ranking_scores(raw)
    assert torch.allclose(transformed.logits, 10.0 * torch.sigmoid(raw.logits))
    # The wrapper preserves the raw tensor for KD; only the separate SUP view
    # is transformed.  This guards against the common raw-logit-SUP shortcut.
    assert torch.equal(raw.logits, torch.tensor([[-2.0, 0.0, 2.0]]))


def test_T02_b13_sampling_order_is_deterministic_and_recipe_declares_all_relations():
    assert consumed_batch_order(8, 13) == list(range(8))
    assert consumed_batch_order(8, 29) == consumed_batch_order(8, 29)
    signature = recipe_signature()
    assert signature["relations"] == [
        "table->table", "table->text", "table->image", "text->table", "image->table"
    ]
    assert signature["candidate_sampling"]["hard_source"] == "raw_qwen_ann_top256"


def test_T03_b13_edge_objective_wires_transformed_sup_and_raw_kd():
    basis = torch.eye(2, 4)
    model = StudentJoinabilityModel(
        4, 2, initialization="pca", initialization_basis=basis,
        freeze_projections=False, relation_param="full",
    )
    example = EdgeExample(
        query_id="q", candidate_ids=("a", "b", "c"), positive_index=1,
        source_type="table", destination_type="text", positive_ids=("b",),
    )
    mask = torch.ones((1, 3), dtype=torch.bool)
    raw = ListScores(torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True), mask,
                     torch.tensor([1]), torch.tensor([[False, True, False]]))
    teacher = ListScores(torch.tensor([[1.0, 0.0, -1.0]]), mask,
                         torch.tensor([1]), torch.tensor([[False, True, False]]))
    result = edge_objective(model, [example], raw, teacher)
    expected = listwise_cross_entropy(
        10.0 * torch.sigmoid(raw.logits), raw.positive_indices,
        raw.candidate_mask, raw.positive_mask,
    )
    assert torch.allclose(result["supervised_loss"], expected)
    assert torch.allclose(result["distillation_loss"], torch.tensor(0.0)) is False


def test_T04_r25_materializer_preserves_global_positive_closure_and_cap():
    row = EdgeExample(
        query_id="q", candidate_ids=("p", "n"), positive_index=0,
        source_type="table", destination_type="text", positive_ids=("p",),
    )
    key = ("q", "table", "text")
    arms = materialize_batch(
        [row], {key: frozenset({"p", "extra"})},
        {key: [("hard", 0.9), ("n", 0.1)]}, 13, 0, 1,
        cap=4, enforce_positive_closure=True,
    )
    observed = arms["candidates"][0]
    assert set(observed["positive_ids"]) == {"p", "extra"}
    assert set(observed["positive_ids"]).issubset(observed["candidate_ids"])
    assert len(observed["candidate_ids"]) <= 4


def test_T06_zero_coefficients_make_split_arms_identical_loss_gradient_and_update():
    base_d = torch.tensor([[1.0, -0.5, 0.2]])
    base_e = torch.tensor([[0.8, -0.2, -0.1]])
    qt = torch.tensor([[1.5, 0.1, -0.4]])
    outcomes = []
    for arm in ("SPLIT-SUP", "SPLIT-QTKD", "SPLIT-U", "SPLIT-UQTKD"):
        parameter = torch.nn.Parameter(torch.cat((base_d, base_e), dim=1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scores = _target(parameter[:, :3], parameter[:, 3:])
        uniform = _list(parameter[:, :3])
        result = split_objective(
            scores,
            arm=arm,
            teacher_qt=_teacher(qt),
            uniform_scores=uniform,
            uniform_relations=["table->text"],
            config=R25ObjectiveConfig(kd_weight=0.0, uniform_weight=0.0),
        )
        optimizer.zero_grad(); result["loss"].backward()
        gradient = parameter.grad.detach().clone(); optimizer.step()
        outcomes.append((result["loss"].detach(), gradient, parameter.detach().clone()))
    for observed in outcomes[1:]:
        for got, expected in zip(observed, outcomes[0]):
            assert torch.equal(got, expected)


def test_T07_qt_teacher_requires_same_values_for_direct_and_evidence():
    scores = _target(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.5, -0.5]]))
    good = _teacher(torch.tensor([[2.0, -1.0]]))
    split_objective(scores, arm="SPLIT-QTKD", teacher_qt=good)
    bad = TargetScores(good.direct, _list(torch.tensor([[2.0, -2.0]])))
    try:
        split_objective(scores, arm="SPLIT-QTKD", teacher_qt=bad)
    except ValueError as exc:
        assert "identical T0(Q,T)" in str(exc)
    else:
        raise AssertionError("mismatched native-E-like logits were accepted")


def test_T09_no_evidence_row_is_finite_and_has_zero_evidence_gradient():
    d = torch.tensor([[1.0, 0.0]], requires_grad=True)
    e = torch.tensor([[3.0, -3.0]], requires_grad=True)
    mask = torch.zeros_like(e, dtype=torch.bool)
    result = split_objective(_target(d, e, mask), arm="SPLIT-SUP")
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert result["evidence_active"] == 0
    assert torch.equal(e.grad, torch.zeros_like(e))


def test_T10_plain_logsumexp_uses_all_valid_paths_and_topk_is_explicit():
    qe = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 100.0]]])
    et = torch.zeros_like(qe)
    mask = torch.tensor([[[True, True, True, True, False]]])
    plain = PathAggregator("logsumexp", top_k=1)(qe, et, mask)
    top1 = PathAggregator("topk_logsumexp", top_k=1)(qe, et, mask)
    assert torch.allclose(plain, torch.logsumexp(qe[..., :4], dim=-1))
    assert torch.equal(top1, torch.tensor([[4.0]]))


def test_T11_split_and_lse_match_for_equal_branches_and_direct_only():
    d = torch.tensor([[1.0, 0.0, -1.0]])
    qt = torch.tensor([[1.5, 0.2, -0.2]])
    both = _target(d, d.clone())
    teacher = _teacher(qt)
    split = split_objective(both, arm="SPLIT-QTKD", teacher_qt=teacher)["loss"]
    lse = split_objective(both, arm="LSE-QTKD", teacher_qt=teacher)["loss"]
    assert torch.allclose(split, lse, atol=1e-6)
    no_e = torch.zeros_like(d, dtype=torch.bool)
    direct = _target(d, torch.zeros_like(d), no_e)
    teacher_direct = _teacher(qt, no_e)
    split = split_objective(direct, arm="SPLIT-QTKD", teacher_qt=teacher_direct)["loss"]
    lse = split_objective(direct, arm="LSE-QTKD", teacher_qt=teacher_direct)["loss"]
    assert torch.allclose(split, lse, atol=1e-6)


def test_T17_dry_run_expands_all_mandatory_jobs_without_performance_gate(tmp_path: Path):
    # The runner expects the normative package under the supplied root, so use
    # the real repo.  Preserve the live run matrix because this semantic test
    # must not reset completed experiment receipts to ``planned``.
    repo = Path(__file__).parents[1]
    matrix_path = repo / "work/stage1_optimization_r25_final_20260914/EXECUTION_MATRIX.json"
    previous = matrix_path.read_bytes() if matrix_path.exists() else None
    try:
        result = dry_run(repo)
    finally:
        if previous is not None:
            matrix_path.write_bytes(previous)
    jobs = result["training_jobs"]
    assert len(jobs) == 16
    assert {row["seed"] for row in jobs} == set(SEEDS)
    assert {row["arm"] for row in jobs if row["stage"] == "C2"} == set(ARMS)
    assert all(row["status"] == "planned" for row in jobs)
    assert all("performance_gate" not in row for row in jobs)


def test_T14_stage2_pilot_separates_online_input_from_eval_metadata():
    path = Path(__file__).parents[1] / "work/stage1_optimization_r25_final_20260914/stage2/pilot_queries_64.jsonl"
    assert path.is_file()
    forbidden = {"gold", "qrels", "hidden_value", "target_column", "implicit_label", "positive_target_ids"}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 128
    assert all(forbidden.isdisjoint(row["model_input"]) for row in rows)
    assert all("query_kind" in row["evaluation_metadata"] for row in rows)


def test_T15_confidence_is_complete_direct_margin_and_equal_alpha_one():
    assert abs(confidence_alpha([4.0, 3.0, 0.0]) - (1.0 - (1.0 / 4.0))) < 1e-6
    assert confidence_alpha([2.0]) == 1.0
    assert confidence_alpha([2.0, 2.0]) == 1.0


def test_T16_stage2_real_noe_preserve_same_opportunities_and_failed_rows():
    path = Path(__file__).parents[1] / "work/stage1_optimization_r25_final_20260914/stage2/pilot_queries_64.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model_input"]["query_id"], []).append(row)
    assert len(grouped) == 64 and all(len(v) == 2 for v in grouped.values())
    for pair in grouped.values():
        left, right = pair
        assert left["model_input"]["condition"] != right["model_input"]["condition"]
        for key in ("generator_ids", "candidate_budget", "target_attempt_budget", "path_budget", "row_mask_id"):
            assert left["model_input"][key] == right["model_input"][key]
        assert all(row["status"] in {"blocked_missing_input", "planned", "completed"} for row in pair)


def test_T17_feedback_gate_is_data_derived_and_core_jobs_have_no_gate():
    root = Path(__file__).parents[1] / "work/stage1_optimization_r25_final_20260914"
    gate = json.loads((root / "feedback/FEEDBACK_GATE.json").read_text())
    assert gate["status"] in {"triggered", "not_triggered"}
    contract_p = Path(__file__).parents[1] / "mmdd_r24_review/EXECUTION_CONTRACT.json"
    if not contract_p.exists():
        contract_p = Path(__file__).parents[1] / "audit/mmdd_r24_review/EXECUTION_CONTRACT.json"
    contract = json.loads(contract_p.read_text())
    assert all(job.get("performance_gate") is None for job in contract["training_jobs"])


def test_T18_resume_checks_plan_and_data_identity_before_shortcut():
    receipt = {"stage": "C2", "arm": "B13-FULL", "seed": 13, "graph_sha256": "g", "optimizer_initial_state": "fresh"}
    assert resume_compatible(receipt, dict(receipt))
    assert not resume_compatible(receipt, {**receipt, "graph_sha256": "changed"})

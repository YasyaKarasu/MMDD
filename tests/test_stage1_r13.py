from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import TargetCandidate, TargetExample
from mmdd_stage1.features import ObjectFeatures
from mmdd_stage1.models import StudentJoinabilityModel, split_table_projection
from mmdd_stage1.scoring import ListScores, TargetScores
from mmdd_stage1.training import checkpoint, student_anchor_loss
from mmdd_stage1.witness_supervision import witness_auxiliary_loss
from run_stage1_r13 import _recall_record


def _feature(object_id: str, object_type: str, values: list[float]) -> ObjectFeatures:
    return ObjectFeatures(object_id, object_type, torch.tensor(values))


def test_split_migration_is_step_zero_equivalent_for_all_deployed_relations():
    torch.manual_seed(13)
    shared = StudentJoinabilityModel(4, 4, initialization="identity")
    with torch.no_grad():
        for parameter in shared.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    shared.reset_projection_anchors()
    split = split_table_projection(shared)
    objects = {
        "q": _feature("same-table", "table", [1.0, 2.0, 3.0, 4.0]),
        "t": _feature("same-table", "table", [1.0, 2.0, 3.0, 4.0]),
        "x": _feature("text", "text", [0.5, -1.0, 2.0, 0.25]),
        "i": _feature("image", "image", [2.0, 0.0, -0.5, 1.0]),
    }
    for source_id, destination_id in (
        ("q", "t"),
        ("q", "x"),
        ("q", "i"),
        ("x", "t"),
        ("i", "t"),
    ):
        source = objects[source_id]
        destination = objects[destination_id]
        torch.testing.assert_close(
            split.score_pairs([source], [destination]),
            shared.score_pairs([source], [destination]),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            split.relation_query(
                source.embedding,
                source.object_type,
                destination.object_type,
                source_role="query" if source.object_type == "table" else None,
            )
            @ split.index_vector(
                destination.embedding,
                destination.object_type,
                destination_role=(
                    "target" if destination.object_type == "table" else None
                ),
            ),
            split.score_pairs([source], [destination])[0],
        )


def test_split_uses_runtime_role_even_when_table_id_is_identical():
    shared = StudentJoinabilityModel(2, 2, initialization="identity")
    split = split_table_projection(shared)
    with torch.no_grad():
        split.projections["table_query"].weight.copy_(
            torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        )
        split.projections["table_target"].weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 3.0]])
        )
    table = _feature("same-id", "table", [1.0, 1.0])
    score = split.score_pairs([table], [table])[0]
    torch.testing.assert_close(score, torch.tensor(5.0))
    assert not torch.equal(
        split.project(table.embedding, "table", role="query"),
        split.project(table.embedding, "table", role="target"),
    )


def test_split_anchor_averages_table_roles_and_checkpoint_round_trips(tmp_path):
    shared = StudentJoinabilityModel(2, 2, initialization="identity")
    split = split_table_projection(shared)
    with torch.no_grad():
        split.projections["table_query"].weight.add_(1.0)
        split.projections["table_target"].weight.add_(1.0)
    shared_with_same_table_drift = StudentJoinabilityModel(
        2, 2, initialization="identity"
    )
    with torch.no_grad():
        shared_with_same_table_drift.projections["table"].weight.add_(1.0)
    torch.testing.assert_close(
        student_anchor_loss(split),
        student_anchor_loss(shared_with_same_table_drift),
    )

    path = tmp_path / "split.pt"
    torch.save(checkpoint(split, "student-edge"), path)
    loaded = load_student(path, torch.device("cpu"))
    assert loaded.projection_mode == "split"
    assert loaded.projection_keys == (
        "table_query",
        "table_target",
        "text",
        "image",
    )
    for key in split.projection_keys:
        torch.testing.assert_close(
            loaded.projections[key].weight, split.projections[key].weight
        )


def _target_scores(path_logits):
    logits = torch.zeros((1, 2), requires_grad=True)
    lists = ListScores(
        logits,
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros(1, dtype=torch.long),
    )
    return TargetScores(lists, lists, (path_logits,))


def test_witness_loss_uses_rows_and_deduplicates_content():
    example = TargetExample(
        "q",
        (
            TargetCandidate("positive", ("e1", "e1-copy", "e2")),
            TargetCandidate("unknown", ("n1",)),
        ),
        0,
        0,
        positive_target_ids=("positive",),
        positive_evidence_by_target={"positive": ("e1", "e1-copy", "e2")},
        positive_evidence_rows_by_target={
            "positive": {"e1": (0,), "e1-copy": (0,), "e2": (1,)}
        },
    )
    positive_paths = torch.tensor([2.0, 2.0, 1.0], requires_grad=True)
    negative_paths = torch.tensor([0.0], requires_grad=True)
    loss, stats = witness_auxiliary_loss(
        [example],
        _target_scores((positive_paths, negative_paths)),
        content_keys={"e1": "same", "e1-copy": "same", "e2": "other"},
    )
    expected = (
        torch.nn.functional.softplus(torch.tensor(-2.0))
        + torch.nn.functional.softplus(torch.tensor(-1.0))
    ) / 2
    torch.testing.assert_close(loss, expected)
    assert stats["eligible_queries"] == 1
    assert stats["eligible_pairs"] == 1
    assert stats["eligible_rows"] == 2
    assert stats["supported_path_ids"] == 3
    assert stats["supported_content_groups"] == 2
    loss.backward()
    assert positive_paths.grad is not None


def test_witness_loss_returns_differentiable_zero_without_eligible_groups():
    example = TargetExample(
        "q",
        (TargetCandidate("positive", ()), TargetCandidate("unknown", ())),
        0,
        0,
        positive_target_ids=("positive",),
    )
    scores = _target_scores((torch.empty(0), torch.empty(0)))
    loss, stats = witness_auxiliary_loss([example], scores)
    assert loss.item() == 0
    assert stats["eligible_queries"] == 0
    loss.backward()


def test_r13_recall_record_deduplicates_targets_and_counts_actual_search_vectors():
    record = {
        "query_id": "q",
        "query_kind": "implicit",
        "positive_target_ids": ["t1", "t2"],
        "paths_by_target": {
            "t1": [
                {"kind": "direct"},
                {"kind": "evidence", "evidence_id": "e1"},
            ],
            "x": [{"kind": "evidence", "evidence_id": "e2"}],
        },
    }
    ranking = [
        {"target_id": "t1"},
        {"target_id": "t1"},
        {"target_id": "x"},
        {"target_id": "t2"},
    ]
    result = _recall_record(record, ranking, ranking, ranking)
    top10 = result["rankings"]["f1_union_direct"]["10"]
    assert top10["target_ids"] == ["t1", "x", "t2"]
    assert top10["numerator"] == 2
    assert top10["denominator"] == 2
    assert top10["recall"] == 1.0
    assert result["search_vectors"] == 5


def test_r13_recall_record_keeps_missed_query_and_ignores_witness_gt_fields():
    ranking = [{"target_id": "x"}]
    base = {
        "query_id": "q",
        "query_kind": "implicit",
        "positive_target_ids": ["t"],
        "paths_by_target": {"x": [{"kind": "direct"}]},
    }
    with_gt = dict(
        base,
        positive_evidence_by_target={"t": ["privileged-evidence"]},
        positive_evidence_rows_by_target={"t": {"privileged-evidence": [3]}},
    )
    left = _recall_record(base, ranking, ranking, ranking)
    right = _recall_record(with_gt, ranking, ranking, ranking)
    assert left == right
    assert left["rankings"]["f1_union_direct"]["10"]["recall"] == 0.0


def test_fixed_unique_candidate_set_preserves_recall_at_n_under_reranking():
    positives = {"b", "d"}
    before = ["a", "b", "c", "d"]
    after = ["d", "c", "a", "b"]
    assert set(before) == set(after)
    assert len(positives & set(before)) / len(positives) == len(
        positives & set(after)
    ) / len(positives)

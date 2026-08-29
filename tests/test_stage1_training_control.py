from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import refresh_stage1_hard_negatives
import run_stage1_rounds
import diagnose_stage1_teacher_rerank
import train_stage1
from mmdd_stage1.data import EdgeExample, TargetCandidate, TargetExample
from mmdd_stage1.evaluation import evaluate_student_retrieval
from mmdd_stage1.models import TeacherJoinabilityModel
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.protocol import validate_protocol_split
from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.selection import (
    CheckpointManager,
    MetricGate,
    validate_stage2_gate,
    write_json,
)
from mmdd_stage1.training import oversample_student_edges, sample_mixed_epoch
from mmdd_stage1.workflow import RoundStep, validate_round_index, workflow_fingerprint


def test_checkpoint_gate_keeps_best_and_last_separate_and_stops_on_patience(tmp_path):
    manager = CheckpointManager(tmp_path / "student.pt")
    gate = MetricGate("recall@10", patience=2)

    first = manager.save_candidate(1, {"epoch": 1})
    first_decision = gate.observe(1, {"recall@10": 0.5})
    manager.update_best(first)
    second = manager.save_candidate(2, {"epoch": 2})
    second_decision = gate.observe(2, {"recall@10": 0.4})
    third = manager.save_candidate(3, {"epoch": 3})
    third_decision = gate.observe(3, {"recall@10": 0.3})

    assert first_decision.improved
    assert not second_decision.improved
    assert not second_decision.should_stop
    assert third_decision.should_stop
    assert gate.best_epoch == 1
    assert torch.load(manager.paths["best"], weights_only=True)["epoch"] == 1
    assert torch.load(manager.paths["last"], weights_only=True)["epoch"] == 3
    assert first != second != third


def test_teacher_rerank_diagnostic_correlation_and_decision_thresholds():
    assert diagnose_stage1_teacher_rerank.spearman_correlation(
        [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]
    ) == pytest.approx(-1.0)
    assert (
        diagnose_stage1_teacher_rerank._branch(0.36, 0.39)
        == "teacher_adds_retrieval_value"
    )
    assert (
        diagnose_stage1_teacher_rerank._branch(0.36, 0.35)
        == "teacher_has_no_incremental_value"
    )
    assert (
        diagnose_stage1_teacher_rerank._branch(0.36, 0.33)
        == "teacher_is_harmful"
    )


def test_teacher_rerank_interval_must_fit_training_schedule():
    args = argparse.Namespace(
        stage="teacher-path",
        teacher_rerank=True,
        teacher_rerank_interval=3,
        epochs=2,
    )

    with pytest.raises(
        ValueError, match="--teacher-rerank-interval cannot exceed --epochs"
    ):
        train_stage1._validate_teacher_rerank_interval(args)


def test_teacher_rerank_interval_saves_but_does_not_gate_skipped_epoch(
    tmp_path, monkeypatch
):
    controller = object.__new__(train_stage1._EpochController)
    controller.stage = "teacher-path"
    controller.aggregator = PathAggregator("logsumexp", 4)
    controller.manager = CheckpointManager(tmp_path / "teacher.pt")
    controller.gate = MetricGate("teacher_rerank.recall@10", patience=3)
    controller.args = argparse.Namespace(
        teacher_rerank=True,
        teacher_rerank_interval=2,
    )
    model = TeacherJoinabilityModel(
        input_dim=4,
        model_dim=4,
        num_heads=1,
        num_layers=1,
        text_latents=1,
        image_latents=1,
        dropout=0.0,
    )
    monkeypatch.setattr(
        train_stage1,
        "evaluate_teacher_reranking",
        lambda *_args, **_kwargs: pytest.fail("rerank should be skipped"),
    )
    record = {"dev_loss": 0.5}

    assert controller(1, model, record) is False

    assert record["teacher_rerank_skipped"] == {"interval": 2, "next_epoch": 2}
    assert controller.gate.best_value is None
    assert controller.manager.paths["last"].is_file()
    assert not controller.manager.paths["best"].exists()


def test_base_hard_sampling_is_explicit_proportional_and_deterministic():
    base = [
        EdgeExample(f"base_{index}", ("positive", "negative"), 0, dataset="base")
        for index in range(4)
    ]
    hard = [
        EdgeExample(f"hard_{index}", ("positive", "negative"), 0, dataset="hard")
        for index in range(2)
    ]

    sampled, counts = sample_mixed_epoch(
        base,
        hard,
        random.Random(17),
        hard_fraction=0.5,
        dataset_sampling_alpha=1.0,
    )
    repeated, repeated_counts = sample_mixed_epoch(
        base,
        hard,
        random.Random(17),
        hard_fraction=0.5,
        dataset_sampling_alpha=1.0,
    )

    assert counts == repeated_counts == {"base": 4, "hard": 4}
    assert [example.query_id for example in sampled] == [
        example.query_id for example in repeated
    ]
    assert sum(example.query_id.startswith("hard_") for example in sampled) == 4


def test_student_edge_type_oversampling_repeats_only_requested_direction():
    examples = [
        EdgeExample(
            "text_query",
            ("positive", "negative"),
            0,
            source_type="text",
            destination_type="table",
        ),
        EdgeExample(
            "table_query",
            ("positive", "negative"),
            0,
            source_type="table",
            destination_type="text",
        ),
    ]

    sampled = oversample_student_edges(
        examples, {"text_table": 3}, random.Random(7)
    )

    assert sum(example.query_id == "text_query" for example in sampled) == 3
    assert sum(example.query_id == "table_query" for example in sampled) == 1


def test_student_path_can_start_from_fresh_frozen_pca(tmp_path):
    basis = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    artifact = tmp_path / "pca.pt"
    torch.save(
        {
            "format_version": 1,
            "input_dim": 4,
            "student_dim": 2,
            "projection": basis,
        },
        artifact,
    )
    args = argparse.Namespace(
        student_checkpoint=None,
        student_initialization="pca",
        student_pca_basis=str(artifact),
        student_dim=2,
        student_init_noise_std=0.01,
        freeze_projection=True,
    )

    student, source = train_stage1._load_or_initialize_student(
        args, embedding_dim=4, device=torch.device("cpu")
    )

    assert source == "fresh_initialization"
    assert student.freeze_projections
    for projection in student.projections.values():
        torch.testing.assert_close(projection.weight, basis)
        assert not projection.weight.requires_grad


def test_full_corpus_metrics_include_fused_direct_evidence_and_path_coverage():
    class StaticIndices:
        def search(self, source_id, destination_type, k):
            values = {
                ("q", "table"): [("wrong", 2.0), ("positive", 1.0)],
                ("q", "text"): [("positive_evidence", 2.0)],
                ("q", "image"): [],
                ("positive_evidence", "table"): [("positive", 2.0)],
            }
            return values.get((source_id, destination_type), [])[:k]

        def search_many(self, source_ids, destination_type, k):
            return [
                self.search(source_id, destination_type, k)
                for source_id in source_ids
            ]

    example = TargetExample(
        "q",
        (
            TargetCandidate("positive", ("positive_evidence",)),
            TargetCandidate("wrong", ()),
        ),
        direct_positive_index=0,
        evidence_positive_index=0,
        split="dev",
        positive_target_ids=("positive",),
        dataset="EntiTables",
    )

    metrics = evaluate_student_retrieval(
        [example], StaticIndices(), evidence_types=("text",)
    )

    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@100"] == 1.0
    assert metrics["direct"]["recall@1"] == 0.0
    assert metrics["direct"]["mrr@100"] == pytest.approx(0.5)
    assert metrics["evidence"]["recall@1"] == 1.0
    assert metrics["positive_evidence_path_queries@10"] == 1
    assert metrics["positive_evidence_path_coverage@10"] == 1.0
    assert metrics["by_dataset"]["EntiTables"]["queries"] == 1
    assert metrics["by_dataset"]["EntiTables"]["direct"]["recall@10"] == 1.0


def test_teacher_feature_readiness_requires_actual_hidden_states(tmp_path):
    data = tmp_path / "edges.jsonl"
    data.write_text(
        json.dumps(
            {
                "query_id": "q",
                "candidate_ids": ["t"],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    features = tmp_path / "features.pt"
    objects = {
        object_id: {
            "object_type": "table",
            "embedding": torch.ones(4),
        }
        for object_id in ("q", "t")
    }
    torch.save({"objects": objects}, features)

    assert not run_stage1_rounds._teacher_features_ready(features, [data])

    for payload in objects.values():
        payload["hidden_states"] = torch.ones(2, 4)
    torch.save({"objects": objects}, features)
    assert run_stage1_rounds._teacher_features_ready(features, [data])


def test_round_index_rejects_old_checkpoint_or_corpus(tmp_path):
    index_dir = tmp_path / "round_01" / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "table.hnsw").write_bytes(b"index")
    (index_dir / "table_ids.json").write_text("[]\n", encoding="utf-8")
    write_json(
        index_dir / "manifest.json",
        {
            "student_checkpoint_sha256": "student-a",
            "corpus_sha256": "corpus-a",
            "types": {
                "table": {
                    "index_path": "table.hnsw",
                    "ids_path": "table_ids.json",
                }
            },
        },
    )

    validate_round_index(
        index_dir,
        student_checkpoint_sha256="student-a",
        corpus_sha256="corpus-a",
    )
    with pytest.raises(ValueError, match="different Student"):
        validate_round_index(
            index_dir,
            student_checkpoint_sha256="student-b",
            corpus_sha256="corpus-a",
        )
    with pytest.raises(ValueError, match="different corpus"):
        validate_round_index(
            index_dir,
            student_checkpoint_sha256="student-a",
            corpus_sha256="corpus-b",
        )


def test_round_step_resumes_only_with_matching_fingerprint(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    fingerprint = workflow_fingerprint({"round": 1}, [source])
    marker = tmp_path / "step.json"
    step = RoundStep(marker, fingerprint)
    step.complete([output])

    assert step.completed()
    stale = RoundStep(marker, workflow_fingerprint({"round": 2}, [source]))
    with pytest.raises(ValueError, match="different fingerprint"):
        stale.completed()


def _stage1_gate(tmp_path: Path, *, allowed: bool) -> tuple[Path, Path, str]:
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"student")
    sha256 = checkpoint_fingerprint(checkpoint)
    gate = tmp_path / "selection.json"
    write_json(
        gate,
        {
            "format_version": 1,
            "completed_stage": "student-path",
            "selection_split": "dev",
            "stage2_allowed": allowed,
            "best_checkpoint": str(checkpoint),
            "best_checkpoint_sha256": sha256,
            "best_metrics": {"positive_evidence_path_coverage@10": 0.0},
        },
    )
    return gate, checkpoint, sha256


def test_stage2_gate_blocks_low_coverage_and_wrong_retrieval_checkpoint(tmp_path):
    blocked_gate, _checkpoint, _sha256 = _stage1_gate(tmp_path, allowed=False)
    with pytest.raises(ValueError, match="insufficient positive evidence-path"):
        validate_stage2_gate(blocked_gate)

    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    gate, _checkpoint, sha256 = _stage1_gate(allowed_dir, allowed=True)
    retrieval = allowed_dir / "retrieval.jsonl"
    retrieval.write_text(
        json.dumps({"query_id": "q", "student_checkpoint_sha256": sha256}) + "\n",
        encoding="utf-8",
    )
    assert validate_stage2_gate(gate, [retrieval])["stage2_allowed"]
    retrieval.write_text(
        json.dumps({"query_id": "q", "student_checkpoint_sha256": "old"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dev-gated best"):
        validate_stage2_gate(gate, [retrieval])


def test_test_split_cannot_be_used_for_dev_selection_or_mining():
    with pytest.raises(ValueError, match="dev_gate must use split 'dev'"):
        validate_protocol_split("dev_gate", "test")

    args = argparse.Namespace(
        mine_only=False,
        output_edge_lists=None,
        hard_targets_per_query=1,
        teacher_checkpoint="teacher.pt",
        teacher_batch_size=1,
        hard_evidence_per_type=0,
        hard_paths_per_query=0,
        split="test",
    )
    with pytest.raises(ValueError, match="mining must use split 'train'"):
        refresh_stage1_hard_negatives.run(args)

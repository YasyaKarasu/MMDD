from __future__ import annotations

import argparse
import json
import random
import statistics
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
import mmdd_stage1.evaluation as evaluation_module
from mmdd_stage1.data import EdgeExample, TargetCandidate, TargetExample
from mmdd_stage1.evaluation import (
    evaluate_direct_retrieval,
    evaluate_student_retrieval,
    retrieval_metric_values,
)
from mmdd_stage1.models import StudentJoinabilityModel, TeacherJoinabilityModel
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.protocol import validate_protocol_split
from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.significance import paired_bootstrap_delta
from mmdd_stage1.selection import (
    CheckpointManager,
    MetricGate,
    validate_stage2_gate,
    write_json,
)
from mmdd_stage1.training import oversample_student_edges, sample_mixed_epoch
from mmdd_stage1.teacher_rerank import (
    _retrieval_metrics as teacher_retrieval_metrics,
    ensemble_scores,
    z_scores,
)
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

    manager.prune_candidates({1, 3})
    assert first.is_file()
    assert not second.exists()
    assert third.is_file()


def test_per_dataset_gate_parses_alias_and_evaluates_nested_metric():
    constraint = train_stage1._parse_per_dataset_gate(
        "wdc2k_v2:direct_recall@10>=0.609"
    )

    results = train_stage1._per_dataset_gate_results(
        {"by_dataset": {"wdc2k_v2": {"direct": {"recall@10": 0.61}}}},
        [constraint],
    )

    assert constraint == ("wdc2k_v2", "direct.recall@10", ">=", 0.609)
    assert results[0]["satisfied"] is True


def test_per_dataset_gate_rejects_invalid_syntax():
    with pytest.raises(argparse.ArgumentTypeError, match="DATASET:METRIC"):
        train_stage1._parse_per_dataset_gate("wdc2k_v2=0.609")


def test_explicit_zero_relation_learning_rate_is_preserved():
    student = StudentJoinabilityModel(
        input_dim=4,
        student_dim=2,
        initialization="pca",
        initialization_basis=torch.eye(2, 4),
    )

    assert train_stage1._student_relation_learning_rate(
        student,
        configured=0.0,
        projection_learning_rate=1e-5,
    ) == 0.0


def test_training_validation_accepts_zero_relation_learning_rate(tmp_path):
    args = train_stage1._argument_parser().parse_args(
        [
            "student-edge",
            "--features",
            str(tmp_path / "missing_features"),
            "--base-data",
            str(tmp_path / "missing_edges.jsonl"),
            "--dev-data",
            str(tmp_path / "missing_edges.jsonl"),
            "--output",
            str(tmp_path / "student.pt"),
            "--relation-learning-rate",
            "0",
            "--kd-target-teacher-alpha",
            "1",
        ]
    )

    with pytest.raises(FileNotFoundError):
        train_stage1.run(args)


def test_paired_bootstrap_resamples_query_pairs_and_reports_ci():
    same = paired_bootstrap_delta([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], iterations=500, seed=7)
    assert same["mean"] == 0.0
    assert same["ci_low"] == 0.0
    assert same["ci_high"] == 0.0

    improved = paired_bootstrap_delta([1.0, 1.0, 0.0], [0.0, 0.0, 0.0], iterations=500, seed=7)
    assert improved["mean"] == pytest.approx(2 / 3)
    assert improved["ci_low"] >= 0.0
    assert improved["p_delta_lt_0"] == 0.0


def test_per_dataset_gate_uses_paired_ci_against_raw_baseline():
    constraint = train_stage1._parse_per_dataset_gate(
        "wdc2k_v2:direct_recall@10>=0.609"
    )
    metrics = {
        "by_dataset": {
            "wdc2k_v2": {
                "direct": {"recall@10": 0.50},
                "per_query": {"direct": {"recall@10": [1.0, 0.0, 1.0]}},
            }
        },
        "raw_embedding": {
            "by_dataset": {
                "wdc2k_v2": {
                    "per_query": {"direct": {"recall@10": [1.0, 0.0, 1.0]}}
                }
            }
        },
    }

    result = train_stage1._per_dataset_gate_results(
        metrics, [constraint], bootstrap_iterations=500
    )[0]

    assert result["gate_mode"] == "paired_bootstrap_ci"
    assert result["satisfied"] is True
    assert result["bootstrap"]["ci_low"] == 0.0


def test_per_dataset_gate_falls_back_to_epoch_zero_when_never_satisfied(tmp_path):
    controller = object.__new__(train_stage1._EpochController)
    controller.manager = CheckpointManager(tmp_path / "student.pt")
    candidate = controller.manager.save_candidate(0, {"epoch": 0})
    controller.gate = MetricGate("recall@10")
    controller.best_metrics = None
    controller.best_index = None
    controller.per_dataset_gates = [
        ("wdc2k_v2", "direct.recall@10", ">=", 0.609)
    ]
    controller.epoch_zero_fallback = (candidate, {"recall@10": 0.35}, None)
    controller.gate_unsatisfied = False

    controller.finalize_gate()

    assert controller.gate_unsatisfied is True
    assert controller.gate.best_epoch == 0
    assert controller.best_metrics == {"recall@10": 0.35}
    assert torch.load(controller.manager.paths["best"], weights_only=True) == {
        "epoch": 0
    }


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


def test_teacher_ensemble_normalizes_within_query_and_preserves_alpha_endpoints():
    raw = [1.0, 2.0, 3.0]
    teacher = [30.0, 20.0, 10.0]

    assert z_scores([4.0, 4.0]) == [0.0, 0.0]
    assert ensemble_scores(raw, teacher, 0.0) == pytest.approx(z_scores(raw))
    assert ensemble_scores(raw, teacher, 1.0) == pytest.approx(z_scores(teacher))
    assert ensemble_scores(raw, teacher, 0.5) == pytest.approx([0.0, 0.0, 0.0])


def test_teacher_ensemble_selection_enforces_wdc_raw_guard():
    payload = {
        "raw_direct": {
            "by_dataset": {"wdc2k_v2": {"recall@10": 0.63}}
        },
        "ensembles": [
            {
                "alpha": 0.7,
                "recall@10": 0.45,
                "mrr@100": 0.2,
                "by_dataset": {"wdc2k_v2": {"recall@10": 0.60}},
            },
            {
                "alpha": 0.3,
                "recall@10": 0.42,
                "mrr@100": 0.3,
                "by_dataset": {"wdc2k_v2": {"recall@10": 0.63}},
            },
        ],
    }

    selected = diagnose_stage1_teacher_rerank._select_ensemble(payload)

    assert selected["alpha"] == pytest.approx(0.3)
    assert selected["accepted"] is True


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
        [example],
        StaticIndices(),
        evidence_types=("text",),
        identity_baseline_metrics={
            "evidence": {"recall@10": 0.75},
            "by_dataset": {
                "EntiTables": {"evidence": {"recall@10": 0.5}}
            },
        },
    )

    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@50"] == 1.0
    assert metrics["direct"]["mrr@50"] == pytest.approx(0.5)
    assert metrics["evidence"]["recall@10"] == 1.0
    assert "recall@100" not in metrics
    assert "mrr@100" not in metrics
    assert metrics["positive_evidence_path_queries@10"] == 1
    assert metrics["positive_evidence_path_coverage@10"] == 1.0
    assert metrics["by_dataset"]["EntiTables"]["queries"] == 1
    assert metrics["by_dataset"]["EntiTables"]["direct"]["recall@10"] == 1.0
    assert metrics["fused_e0"]["positive_evidence_path_coverage@10"] == 1.0
    assert metrics["fused_e005"]["recall@10"] == 1.0
    assert metrics["evidence_identity_baseline"]["recall@10"] == 0.75
    assert metrics["evidence_identity_baseline"]["by_dataset"]["EntiTables"][
        "recall@10"
    ] == 0.5


def test_direct_retrieval_recall_uses_all_positive_targets():
    class StaticIndices:
        @staticmethod
        def search(_source_id, _destination_type, _k):
            return [("positive_1", 2.0), ("wrong", 1.0)]

    example = TargetExample(
        "q",
        (
            TargetCandidate("positive_1", ()),
            TargetCandidate("positive_2", ()),
            TargetCandidate("wrong", ()),
        ),
        direct_positive_index=0,
        evidence_positive_index=0,
        split="dev",
        positive_target_ids=("positive_1", "positive_2"),
    )

    metrics = evaluate_direct_retrieval(
        [example], StaticIndices(), recall_ks=(2,)
    )

    assert metrics["recall@2"] == 0.5
    assert metrics["mrr@2"] == 1.0


def test_shared_retrieval_metric_values_preserve_historical_aggregation():
    positives = [
        {f"p{query}_{index}" for index in range(10)}
        for query in range(3)
    ]
    rankings = [
        [f"p{query}_{index}" for index in range(hit_count)]
        for query, hit_count in enumerate((1, 2, 3))
    ]

    per_query = retrieval_metric_values(rankings, positives, (10,))
    evaluation_metrics = evaluation_module._channel_metrics(
        {10: rankings}, positives, (10,)
    )
    teacher_metrics = teacher_retrieval_metrics(rankings, positives, (10,))

    assert per_query == {
        "recall@10": [0.1, 0.2, 0.3],
        "mrr@10": [1.0, 1.0, 1.0],
    }
    assert evaluation_metrics["recall@10"] == sum(per_query["recall@10"]) / 3
    assert teacher_metrics["recall@10"] == statistics.fmean(
        per_query["recall@10"]
    )


def test_path_coverage_uses_an_independent_k10_pool(monkeypatch):
    calls = []

    def fake_retrieve(query_ids, _indices, *, k, **_kwargs):
        calls.append(k)
        evidence_paths = (
            [{"kind": "evidence", "evidence_id": "e", "path_score": 1.0}]
            if k == 20
            else []
        )
        direct = [
            {
                "target_id": "positive",
                "direct_score": 1.0,
                "evidence_score": None,
                "score": 1.0,
                "paths": [{"kind": "direct", "path_score": 1.0}],
            }
        ]
        evidence = (
            [
                {
                    "target_id": "positive",
                    "direct_score": None,
                    "evidence_score": 1.0,
                    "score": 1.0,
                    "paths": evidence_paths,
                }
            ]
            if evidence_paths
            else []
        )
        fused = [
            {
                "target_id": "positive",
                "direct_score": 1.0,
                "evidence_score": 1.0 if evidence_paths else None,
                "score": 1.0,
                "paths": [*direct[0]["paths"], *evidence_paths],
            }
        ]
        return [{"direct": direct, "evidence": evidence, "fused": fused}]

    monkeypatch.setattr(
        evaluation_module, "retrieve_zero_one_hop_detailed_many", fake_retrieve
    )
    example = TargetExample(
        "q",
        (TargetCandidate("positive", ("e",)),),
        direct_positive_index=0,
        evidence_positive_index=0,
        split="dev",
        positive_target_ids=("positive",),
        dataset="data",
    )

    metrics = evaluate_student_retrieval(
        [example], object(), recall_ks=(20,), gamma=4, gamma_evidence=2
    )

    assert calls == [10, 20]
    assert metrics["recall@20"] == 1.0
    assert metrics["positive_evidence_path_coverage@10"] == 0.0
    assert metrics["retrieval_budget"]["coverage_k"] == 10


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

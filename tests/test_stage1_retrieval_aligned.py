from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.retrieval_aligned import align_edge_record, align_target_record
from mmdd_stage1.data import TargetCandidate, TargetExample
from mmdd_stage1 import teacher_rerank
import audit_stage1_teacher_images
import evaluate_stage1_r3_baselines
import merge_stage1_teacher_cache
import partition_stage1_teacher_work
import summarize_stage1_teacher_run
import sweep_stage1_fusion


class StaticIndices:
    def search(self, source_id, destination_type, k):
        values = {
            "table": [("q", 1.0), ("positive", 0.9), ("raw_t1", 0.8), ("raw_t2", 0.7), ("raw_t3", 0.6)],
            "text": [("raw_text", 0.8)],
            "image": [("raw_image", 0.7)],
        }
        return values[destination_type][:k]


def test_fusion_metrics_preserve_custom_cutoffs_and_dataset_breakdown():
    examples = [
        TargetExample(
            "q1",
            (TargetCandidate("p1", ()),),
            0,
            0,
            dataset="alpha",
            positive_target_ids=("p1",),
        ),
        TargetExample(
            "q2",
            (TargetCandidate("p2", ()),),
            0,
            0,
            dataset="beta",
            positive_target_ids=("p2",),
        ),
    ]

    metrics = sweep_stage1_fusion._with_datasets(
        examples,
        [["p1", "n1"], ["n2", "p2"]],
        [{"p1"}, {"p2"}],
        (1, 2),
        [True, False],
    )

    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr@2"] == 0.75
    assert metrics["positive_evidence_path_coverage@10"] == 0.5
    assert metrics["by_dataset"]["alpha"]["recall@1"] == 1.0
    assert metrics["by_dataset"]["beta"]["recall@1"] == 0.0


def test_r3_baseline_comparison_view_selects_the_requested_path_channel():
    metrics = {
        "queries": 2,
        "per_query": {
            "fused": {"recall@10": [0.0, 1.0]},
            "direct": {"recall@10": [1.0, 0.0]},
        },
        "by_dataset": {
            "data": {
                "queries": 2,
                "per_query": {
                    "fused": {"recall@10": [0.0, 1.0]},
                    "direct": {"recall@10": [1.0, 0.0]},
                },
            }
        },
    }

    fused = evaluate_stage1_r3_baselines._comparison_view(metrics, channel="fused")
    direct = evaluate_stage1_r3_baselines._comparison_view(metrics, channel="direct")
    comparison = evaluate_stage1_r3_baselines._comparison(
        fused,
        direct,
        recall_ks=(10,),
        iterations=10,
        seed=13,
    )

    assert fused["per_query"]["recall@10"] == [0.0, 1.0]
    assert direct["per_query"]["recall@10"] == [1.0, 0.0]
    assert comparison["overall"]["recall@10"]["mean"] == 0.0


def _object_type(object_id: str) -> str:
    return "image" if "image" in object_id else "text"


def test_align_edge_record_keeps_handcrafted_quota_and_adds_raw_negatives():
    record = {
        "query_id": "q",
        "source_type": "table",
        "positive_id": "positive",
        "candidate_ids": ["positive", "hand1", "hand2"],
        "destination_type": "table",
        "dataset": "data",
        "split": "train",
    }

    aligned = align_edge_record(
        record, StaticIndices(), list_width=4, handcrafted_negatives=1
    )

    assert aligned["candidate_ids"] == ["positive", "hand1", "raw_t1", "raw_t2"]
    without_bad = align_edge_record(
        record,
        StaticIndices(),
        list_width=4,
        handcrafted_negatives=1,
        unavailable_ids={"raw_t1"},
    )
    assert without_bad["candidate_ids"] == ["positive", "hand1", "raw_t2", "raw_t3"]


def test_align_target_record_attaches_query_hard_evidence_to_raw_targets():
    record = {
        "query_id": "q",
        "direct_positive_target_id": "positive",
        "evidence_positive_target_id": "positive",
        "positive_target_ids": ["positive"],
        "candidates": [
            {"target_id": "positive", "evidence_ids": ["positive_text", "positive_image"]},
            {"target_id": "hand1", "evidence_ids": ["hand_text"]},
        ],
        "dataset": "data",
        "split": "train",
    }

    aligned = align_target_record(
        record,
        StaticIndices(),
        _object_type,
        list_width=4,
        handcrafted_negatives=1,
        evidence_per_type=1,
    )

    by_target = {value["target_id"]: value for value in aligned["candidates"]}
    assert list(by_target) == ["positive", "hand1", "raw_t1", "raw_t2"]
    assert by_target["positive"]["evidence_ids"] == ["positive_text", "positive_image"]
    assert by_target["raw_t1"]["evidence_ids"] == ["raw_text", "raw_image"]
    assert len(aligned["candidates"]) == 4


def test_align_target_record_can_bind_ann_targets_to_their_own_evidence():
    record = {
        "query_id": "q",
        "direct_positive_target_id": "positive",
        "evidence_positive_target_id": "positive",
        "positive_target_ids": ["positive"],
        "candidates": [
            {"target_id": "positive", "evidence_ids": ["positive_text"]},
            {"target_id": "hand1", "evidence_ids": ["hand_text"]},
        ],
        "dataset": "data",
        "split": "train",
    }

    aligned = align_target_record(
        record,
        StaticIndices(),
        _object_type,
        list_width=4,
        handcrafted_negatives=1,
        evidence_per_type=1,
        evidence_binding="target-bound",
        target_evidence={
            "raw_t1": ["raw_t1_text", "raw_t1_image"],
            "raw_t2": ["raw_t2_text", "raw_t2_image"],
        },
    )

    by_target = {value["target_id"]: value for value in aligned["candidates"]}
    assert by_target["raw_t1"]["evidence_ids"] == [
        "raw_t1_text",
        "raw_t1_image",
    ]
    assert by_target["raw_t2"]["evidence_ids"] == [
        "raw_t2_text",
        "raw_t2_image",
    ]
    assert by_target["raw_t1"]["evidence_ids"] != by_target["raw_t2"][
        "evidence_ids"
    ]


def test_align_target_record_text_only_removes_image_evidence():
    record = {
        "query_id": "q",
        "direct_positive_target_id": "positive",
        "evidence_positive_target_id": "positive",
        "positive_target_ids": ["positive"],
        "candidates": [
            {
                "target_id": "positive",
                "evidence_ids": ["positive_text", "positive_image"],
            }
        ],
        "dataset": "data",
        "split": "train",
    }

    aligned = align_target_record(
        record,
        StaticIndices(),
        _object_type,
        list_width=3,
        handcrafted_negatives=0,
        evidence_per_type=1,
        evidence_types=("text",),
    )

    assert all(
        all("image" not in evidence_id for evidence_id in candidate["evidence_ids"])
        for candidate in aligned["candidates"]
    )


def test_teacher_rerank_metrics_preserve_dataset_breakdown(monkeypatch):
    candidates = [(f"target_{index}", 1.0 - index / 20) for index in range(11)]
    example = TargetExample(
        "q",
        (TargetCandidate("target_0", ()), TargetCandidate("negative", ())),
        0,
        0,
        dataset="data",
        positive_target_ids=("target_0",),
    )
    class TrainingTeacher:
        training = True

        def eval(self):
            self.training = False
            return self

    teacher = TrainingTeacher()

    def scores(model, *_args, **_kwargs):
        assert not model.training
        return [float(index) for index in range(11)]

    monkeypatch.setattr(teacher_rerank, "_teacher_scores", scores)

    metrics = teacher_rerank.evaluate_teacher_reranking(
        teacher,
        [example],
        [candidates],
        object(),
        device=None,
        batch_size=4,
        return_per_query=True,
    )

    assert metrics["raw_direct"]["recall@10"] == 1.0
    assert metrics["teacher_reranked"]["recall@10"] == 0.0
    assert metrics["teacher_reranked"]["by_dataset"]["data"]["queries"] == 1
    assert metrics["raw_direct"]["per_query"]["recall@10"] == [1.0]
    assert metrics["teacher_reranked"]["per_query"]["recall@10"] == [0.0]
    assert not teacher.training


def test_teacher_reranked_indices_preserve_the_fixed_raw_pool(monkeypatch):
    class RawIndices:
        def __init__(self):
            self.calls = []

        def search(self, source_id, destination_type, k):
            self.calls.append((source_id, destination_type, k))
            return [("low", 0.9), ("high", 0.8)][:k]

    class Teacher:
        def eval(self):
            return self

    monkeypatch.setattr(
        teacher_rerank,
        "_teacher_scores",
        lambda _teacher, _source, candidate_ids, *_args: [
            {"low": -1.0, "high": 2.0}[candidate_id]
            for candidate_id in candidate_ids
        ],
    )
    raw = RawIndices()
    indices = teacher_rerank.TeacherRerankedANNIndices(
        raw,
        Teacher(),
        object(),
        device=None,
        batch_size=2,
    )

    hits = indices.search("q", "table", 2)

    assert raw.calls == [("q", "table", 2)]
    assert hits == [("high", 2.0), ("low", -1.0)]


def test_teacher_retrieval_feature_view_uses_explicit_pooled_fallback():
    import torch

    from evaluate_stage1_teacher_retrieval import _TeacherFeatureView
    from mmdd_stage1.features import ObjectFeatures

    class Store:
        def get(self, object_id, *, include_hidden=True):
            return ObjectFeatures(
                object_id=object_id,
                object_type="text",
                embedding=torch.tensor([3.0, 4.0]),
                hidden_states=(
                    torch.tensor([[1.0, 2.0]])
                    if object_id == "cached" and include_hidden
                    else None
                ),
            )

    view = _TeacherFeatureView(Store(), missing_policy="pooled_embedding")

    cached = view.get("cached", include_hidden=True)
    fallback = view.get("missing", include_hidden=True)

    assert cached.hidden_states.tolist() == [[1.0, 2.0]]
    assert fallback.hidden_states.tolist() == [[3.0, 4.0]]
    assert view.missing_ids() == ["missing"]
    assert view.coverage() == {
        "policy": "pooled_embedding",
        "unique_objects_scored": 2,
        "cached_hidden_objects": 1,
        "pooled_embedding_fallback_objects": 1,
        "allowed_pooled_embedding_fallback_objects": 0,
        "unexpected_pooled_embedding_fallback_objects": 1,
        "cached_hidden_fraction": 0.5,
    }


def test_teacher_retrieval_feature_view_only_allows_audited_fallback():
    import torch

    from evaluate_stage1_teacher_retrieval import _TeacherFeatureView
    from mmdd_stage1.features import ObjectFeatures

    class Store:
        def get(self, object_id, *, include_hidden=True):
            return ObjectFeatures(
                object_id=object_id,
                object_type="image",
                embedding=torch.tensor([3.0, 4.0]),
                hidden_states=None,
            )

    view = _TeacherFeatureView(
        Store(), missing_policy="error", allowed_fallback_ids={"invalid"}
    )

    allowed = view.get("invalid", include_hidden=True)
    strict_view = _TeacherFeatureView(
        Store(), missing_policy="error", allowed_fallback_ids={"invalid"}
    )
    disallowed = strict_view.get("unexpected", include_hidden=True)

    assert allowed.hidden_states.tolist() == [[3.0, 4.0]]
    assert disallowed.hidden_states is None
    assert view.allowed_ids_used() == ["invalid"]
    assert view.missing_ids() == []
    assert view.coverage() == {
        "policy": "error",
        "unique_objects_scored": 1,
        "cached_hidden_objects": 0,
        "pooled_embedding_fallback_objects": 1,
        "allowed_pooled_embedding_fallback_objects": 1,
        "unexpected_pooled_embedding_fallback_objects": 0,
        "cached_hidden_fraction": 0.0,
    }


def test_teacher_cache_staging_merge_is_atomic_and_preserves_records(tmp_path):
    cache = tmp_path / "cache"
    staging = tmp_path / "staging"
    (staging / "teacher_objects").mkdir(parents=True)
    record = {
        "object_id": "object",
        "object_type": "text",
        "teacher_feature_path": "teacher_objects/object.pt",
        "source_fingerprint": "fingerprint",
    }
    (staging / "teacher_objects" / "object.pt").write_bytes(b"features")
    (staging / "teacher_manifest.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    cache.mkdir()

    summary = merge_stage1_teacher_cache.run(
        argparse.Namespace(cache_dir=str(cache), staging_dirs=[str(staging)])
    )

    assert summary["teacher_objects_added"] == 1
    assert (cache / "teacher_objects" / "object.pt").read_bytes() == b"features"
    assert json.loads((cache / "teacher_manifest.jsonl").read_text()) == record


def test_teacher_cache_staging_merge_can_move_features(tmp_path):
    cache = tmp_path / "cache"
    staging = tmp_path / "staging"
    (staging / "teacher_objects").mkdir(parents=True)
    record = {
        "object_id": "object",
        "object_type": "text",
        "teacher_feature_path": "teacher_objects/object.pt",
        "source_fingerprint": "fingerprint",
    }
    source = staging / "teacher_objects" / "object.pt"
    source.write_bytes(b"features")
    (staging / "teacher_manifest.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    cache.mkdir()

    summary = merge_stage1_teacher_cache.run(
        argparse.Namespace(
            cache_dir=str(cache),
            staging_dirs=[str(staging)],
            exclude_object_ids=[],
            move=True,
        )
    )

    assert summary["moved"] is True
    assert not source.exists()
    assert (cache / "teacher_objects" / "object.pt").read_bytes() == b"features"


def test_partition_pending_teacher_work_balances_each_modality(tmp_path):
    records = [
        {"object_id": f"{kind}_{index}", "object_type": kind}
        for kind in ("table", "text", "image")
        for index in range(3)
    ]
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    shards = partition_stage1_teacher_work.partition_pending_objects(
        objects,
        {record["object_id"] for record in records},
        {"text_0"},
        2,
    )

    assert sum(len(shard) for shard in shards) == 8
    assert all(record["object_id"] != "text_0" for shard in shards for record in shard)
    for kind in ("table", "text", "image"):
        counts = [sum(record["object_type"] == kind for record in shard) for shard in shards]
        assert max(counts) - min(counts) <= 1


def test_image_audit_preserves_existing_exclusions(tmp_path):
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        json.dumps({"object_id": "selected_text", "object_type": "text"}) + "\n",
        encoding="utf-8",
    )
    selected = tmp_path / "selected.jsonl"
    selected.write_text(json.dumps({"object_id": "selected_text"}) + "\n", encoding="utf-8")
    output = tmp_path / "invalid.jsonl"
    existing = {"object_id": "old_bad_image", "error_type": "OSError"}
    output.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    summary = audit_stage1_teacher_images.run(
        argparse.Namespace(
            input_jsonl=str(objects),
            selected_ids=[str(selected)],
            output=str(output),
        )
    )

    assert summary["new_invalid_images"] == 0
    assert summary["invalid_images"] == 1
    assert json.loads(output.read_text()) == existing


def test_teacher_run_summary_applies_overall_and_dataset_guardrails(tmp_path):
    metrics = {
        "raw_direct": {
            "recall@10": 0.36,
            "by_dataset": {"data": {"recall@10": 0.35}},
        },
        "teacher_reranked": {
            "recall@10": 0.41,
            "recall@100": 0.6,
            "mrr@100": 0.3,
            "by_dataset": {
                "data": {
                    "recall@10": 0.36,
                    "recall@100": 0.55,
                    "mrr@100": 0.25,
                }
            },
        },
    }
    history = tmp_path / "teacher.pt.history.json"
    history.write_text(
        json.dumps(
            {
                "best_epoch": 1,
                "epochs": [
                    {"epoch": 1, "dev_loss": 0.5, "dev_teacher_rerank": metrics}
                ],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_stage1_teacher_run.run(
        argparse.Namespace(
            history=str(history),
            label="Teacher",
            minimum_recall_at_10=0.4,
            output=str(tmp_path / "RESULTS.md"),
            summary=None,
        )
    )

    assert "**PASS**" in report


def test_teacher_run_summary_combines_direct_and_evidence_to_table_gates(tmp_path):
    metrics = {
        "raw_direct": {
            "recall@10": 0.36,
            "by_dataset": {"data": {"recall@10": 0.35}},
        },
        "teacher_reranked": {
            "recall@10": 0.41,
            "recall@100": 0.6,
            "mrr@100": 0.3,
            "by_dataset": {
                "data": {
                    "recall@10": 0.36,
                    "recall@100": 0.55,
                    "mrr@100": 0.25,
                }
            },
        },
    }
    history = tmp_path / "teacher.pt.history.json"
    history.write_text(
        json.dumps(
            {
                "best_epoch": 1,
                "epochs": [
                    {
                        "epoch": 1,
                        "dev_loss": 0.5,
                        "candidate_checkpoint_sha256": "teacher-sha",
                        "dev_teacher_rerank": metrics,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "teacher_checkpoint_sha256": "teacher-sha",
                "raw_direct": metrics["raw_direct"],
                "teacher_reranked": metrics["teacher_reranked"],
                "evidence_to_table": {
                    "raw_direct": {"recall@10": 0.3},
                    "teacher_reranked": {
                        "recall@10": 0.29,
                        "recall@100": 0.6,
                        "mrr@100": 0.2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    gate = tmp_path / "gate.json"

    report = summarize_stage1_teacher_run.run(
        argparse.Namespace(
            history=str(history),
            label="Teacher",
            minimum_recall_at_10=0.4,
            output=str(tmp_path / "RESULTS.md"),
            summary=None,
            diagnostic=str(diagnostic),
            gate_output=str(gate),
        )
    )

    assert "Task 7 combined acceptance: **FAIL**" in report
    gate_payload = json.loads(gate.read_text())
    assert gate_payload["accepted"] is False
    assert gate_payload["direct_pass"] is True


def test_teacher_run_summary_rejects_mixed_direct_metric_versions(tmp_path):
    metrics = {
        "raw_direct": {
            "recall@10": 0.36,
            "by_dataset": {"data": {"recall@10": 0.35}},
        },
        "teacher_reranked": {
            "recall@10": 0.41,
            "recall@100": 0.6,
            "mrr@100": 0.3,
            "by_dataset": {
                "data": {
                    "recall@10": 0.36,
                    "recall@100": 0.55,
                    "mrr@100": 0.25,
                }
            },
        },
    }
    history = tmp_path / "teacher.pt.history.json"
    history.write_text(
        json.dumps(
            {
                "best_epoch": 1,
                "epochs": [
                    {
                        "epoch": 1,
                        "dev_loss": 0.5,
                        "candidate_checkpoint_sha256": "teacher-sha",
                        "dev_teacher_rerank": metrics,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "teacher_checkpoint_sha256": "teacher-sha",
                "raw_direct": metrics["raw_direct"],
                "teacher_reranked": {
                    **metrics["teacher_reranked"],
                    "recall@10": 0.39,
                },
                "evidence_to_table": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="R@10 does not match"):
        summarize_stage1_teacher_run.run(
            argparse.Namespace(
                history=str(history),
                label="Teacher",
                minimum_recall_at_10=0.4,
                output=str(tmp_path / "RESULTS.md"),
                summary=None,
                diagnostic=str(diagnostic),
                gate_output=None,
            )
        )

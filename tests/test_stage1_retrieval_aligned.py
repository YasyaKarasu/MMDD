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
import merge_stage1_teacher_cache
import partition_stage1_teacher_work
import summarize_stage1_teacher_run


class StaticIndices:
    def search(self, source_id, destination_type, k):
        values = {
            "table": [("q", 1.0), ("positive", 0.9), ("raw_t1", 0.8), ("raw_t2", 0.7), ("raw_t3", 0.6)],
            "text": [("raw_text", 0.8)],
            "image": [("raw_image", 0.7)],
        }
        return values[destination_type][:k]


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
    )

    assert metrics["raw_direct"]["recall@10"] == 1.0
    assert metrics["teacher_reranked"]["recall@10"] == 0.0
    assert metrics["teacher_reranked"]["by_dataset"]["data"]["queries"] == 1
    assert not teacher.training


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

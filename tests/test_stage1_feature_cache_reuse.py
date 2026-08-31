from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import merge_stage1_feature_cache
import prepare_stage1_task_f_work
import run_stage1_r6_task_f
import run_stage1_r6_task_f_tokens
import seed_stage1_feature_cache
from cache_stage1_features import _source_fingerprint


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _cache_record(record: dict, directory: str, field: str) -> dict:
    return {
        "object_id": record["object_id"],
        "object_type": record["object_type"],
        field: f"{directory}/{record['object_id']}.pt",
        "source_fingerprint": _source_fingerprint(record),
    }


def test_task_f_work_preserves_non_tables_and_balances_tables(tmp_path: Path):
    text = {"object_id": "text", "object_type": "text", "text": "same"}
    tables = [
        {
            "object_id": f"table_{index}",
            "object_type": "table",
            "embedding_role": "query" if index == 0 else "target",
            "table_parts": ["Columns: A", f"Row: A: {index}"],
        }
        for index in range(3)
    ]
    enhanced = [*tables, text]
    reference = [
        *[{**record, "table_parts": ["Columns: A", "Row: old"]} for record in tables],
        text,
    ]
    lake = tmp_path / "lake.jsonl"
    old = tmp_path / "old.jsonl"
    _write_jsonl(lake, enhanced)
    _write_jsonl(old, reference)

    summary = prepare_stage1_task_f_work.prepare(
        [lake], old, tmp_path / "prepared", num_shards=2
    )

    assert summary["counts"] == {"objects": 4, "tables": 3, "text": 1, "image": 0}
    assert sum(shard["tables"] for shard in summary["shards"]) == 3
    assert max(shard["tables"] for shard in summary["shards"]) <= 2


def test_seed_and_merge_feature_cache_use_hardlinks(tmp_path: Path):
    text = {"object_id": "text", "object_type": "text", "text": "same"}
    table = {
        "object_id": "table",
        "object_type": "table",
        "embedding_role": "target",
        "table_parts": ["Columns: A", "Row: A: one"],
    }
    objects = tmp_path / "objects.jsonl"
    _write_jsonl(objects, [text, table])
    metadata = {"format_version": 5, "model_dir": "model"}

    source = tmp_path / "source"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    text_base = _cache_record(text, "objects", "feature_path")
    text_teacher = _cache_record(text, "teacher_objects", "teacher_feature_path")
    _write_jsonl(source / "manifest.jsonl", [text_base])
    _write_jsonl(source / "teacher_manifest.jsonl", [text_teacher])
    for record, field in ((text_base, "feature_path"), (text_teacher, "teacher_feature_path")):
        path = source / record[field]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"text feature")

    destination = tmp_path / "destination"
    summary = seed_stage1_feature_cache.seed(
        objects, source, destination, object_types={"text", "image"}
    )
    assert summary["base_objects_linked"] == 1
    assert (destination / text_base["feature_path"]).samefile(source / text_base["feature_path"])

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    table_base = _cache_record(table, "objects", "feature_path")
    table_teacher = _cache_record(table, "teacher_objects", "teacher_feature_path")
    _write_jsonl(staging / "manifest.jsonl", [table_base])
    _write_jsonl(staging / "teacher_manifest.jsonl", [table_teacher])
    for record, field in ((table_base, "feature_path"), (table_teacher, "teacher_feature_path")):
        path = staging / record[field]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"table feature")

    merged = merge_stage1_feature_cache.merge(destination, [staging])

    assert merged["manifest.jsonl"]["total"] == 2
    assert merged["teacher_manifest.jsonl"]["total"] == 2
    assert (destination / table_base["feature_path"]).samefile(
        staging / table_base["feature_path"]
    )


def test_seed_can_reuse_table_base_without_reusing_table_teacher(tmp_path: Path):
    table = {
        "object_id": "table",
        "object_type": "table",
        "embedding_role": "target",
        "table_parts": ["Columns: A", "Row: one"],
    }
    objects = tmp_path / "objects.jsonl"
    _write_jsonl(objects, [table])
    source = tmp_path / "source"
    source.mkdir()
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 5,
                "model_dir": "model",
                "table_pooling": "prepooled_schema_rows",
            }
        ),
        encoding="utf-8",
    )
    base = _cache_record(table, "objects", "feature_path")
    teacher = _cache_record(table, "teacher_objects", "teacher_feature_path")
    _write_jsonl(source / "manifest.jsonl", [base])
    _write_jsonl(source / "teacher_manifest.jsonl", [teacher])
    for record, field in ((base, "feature_path"), (teacher, "teacher_feature_path")):
        path = source / record[field]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(field.encode())

    destination = tmp_path / "destination"
    summary = seed_stage1_feature_cache.seed(
        objects,
        source,
        destination,
        object_types={"table"},
        teacher_object_types=set(),
        table_tokens_per_group=4,
    )

    assert summary["base_objects_linked"] == 1
    assert summary["teacher_objects_linked"] == 0
    assert (destination / base["feature_path"]).samefile(source / base["feature_path"])
    assert not (destination / teacher["teacher_feature_path"]).exists()
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["table_tokens_per_group"] == 4
    assert metadata["table_pooling"] == "contiguous_mean_segments"


def test_task_f_summary_compares_evidence_and_teacher_delta(tmp_path: Path):
    def write(name: str, payload: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    enhanced = write(
        "enhanced.json",
        {
            "systems": {
                "student": {
                    "metrics": {
                        "evidence": {"recall@10": 0.75},
                        "per_query": {
                            "evidence": {"recall@10": [1, 1, 1, 0]}
                        },
                    }
                },
                "raw": {
                    "metrics": {
                        "per_query": {"direct": {"recall@10": [1, 0, 0, 0]}}
                    }
                },
                "raw_ensemble": {
                    "metrics": {"per_query": {"recall@10": [1, 1, 0, 0]}}
                },
            }
        },
    )
    baseline_student = write(
        "baseline_student.json",
        {
            "systems": {
                "student": {
                    "metrics": {
                        "evidence": {"recall@10": 0.5},
                        "per_query": {
                            "evidence": {"recall@10": [1, 1, 0, 0]}
                        },
                    }
                }
            }
        },
    )
    baseline_raw = write(
        "baseline_raw.json",
        {
            "systems": {
                "raw": {
                    "metrics": {
                        "per_query": {"direct": {"recall@10": [1, 0, 0, 0]}}
                    }
                }
            }
        },
    )
    baseline_teacher = write(
        "baseline_teacher.json",
        {
            "systems": {
                "raw_ensemble": {
                    "metrics": {"per_query": {"recall@10": [1, 0, 0, 0]}}
                }
            }
        },
    )

    summary = run_stage1_r6_task_f.summarize(
        enhanced,
        baseline_student,
        baseline_raw,
        baseline_teacher,
        tmp_path,
    )

    assert summary["student_evidence_recall@10"]["delta"]["mean"] == 0.25
    assert summary["teacher_rerank_delta@10"]["baseline"] == 0.0
    assert summary["teacher_rerank_delta@10"]["task_f"] == 0.25
    assert summary["teacher_rerank_delta@10"]["difference_in_differences"][
        "mean"
    ] == 0.25


def test_task_f2_summary_keeps_student_fixed_and_compares_teacher_delta(
    tmp_path: Path,
):
    def write(name: str, payload: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    enhanced = write(
        "enhanced_f2.json",
        {
            "systems": {
                "raw": {
                    "metrics": {
                        "per_query": {"direct": {"recall@10": [1, 0, 0, 0]}}
                    }
                },
                "raw_ensemble": {
                    "metrics": {"per_query": {"recall@10": [1, 1, 0, 0]}}
                },
            }
        },
    )
    baseline_student = write(
        "student_f2.json",
        {
            "systems": {
                "student": {
                    "metrics": {
                        "evidence": {"recall@10": 0.5},
                        "per_query": {"evidence": {"recall@10": [1, 1, 0, 0]}},
                    }
                }
            }
        },
    )
    baseline_raw = write(
        "raw_f2.json",
        {
            "systems": {
                "raw": {
                    "metrics": {
                        "per_query": {"direct": {"recall@10": [1, 0, 0, 0]}}
                    }
                }
            }
        },
    )
    baseline_teacher = write(
        "teacher_f2.json",
        {
            "systems": {
                "raw_ensemble": {
                    "metrics": {"per_query": {"recall@10": [1, 0, 0, 0]}}
                }
            }
        },
    )

    summary = run_stage1_r6_task_f_tokens.summarize(
        enhanced,
        baseline_student,
        baseline_raw,
        baseline_teacher,
        tmp_path / "summary",
        table_tokens_per_group=4,
    )

    assert summary["student_evidence_recall@10"]["task_f"] == 0.5
    assert summary["student_evidence_recall@10"]["unchanged_by_construction"] is True
    assert summary["student_evidence_recall@10"]["delta"]["mean"] == 0.0
    assert summary["teacher_rerank_delta@10"]["baseline"] == 0.0
    assert summary["teacher_rerank_delta@10"]["task_f"] == 0.25
    assert summary["teacher_rerank_delta@10"]["difference_in_differences"][
        "mean"
    ] == 0.25

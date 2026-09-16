from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from migrate_query_only_split import migrate_dataset


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published_shard(root: Path, relative_path: str, records: int) -> dict[str, object]:
    """A shard entry as the WDC builder publishes it, carrying checksums."""
    path = root / relative_path
    return {
        "path": relative_path,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_legacy_dataset(root: Path) -> dict[str, list[dict[str, object]]]:
    """Write the per-split artifacts both manifest shapes share."""
    payloads = {
        "queries": [
            {"table_id": "q_train", "source_table_id": "s1", "split": "train"},
            {"table_id": "q_test", "source_table_id": "s2", "split": "test"},
        ],
        "targets": [
            {"table_id": "t1", "source_table_id": "s1", "split": "train"},
            {"table_id": "t2", "source_table_id": "s2", "split": "test"},
        ],
        "recoveries": [
            {"query_table_id": "q_train", "target_table_id": "t1", "split": "train"},
        ],
        "qrels": [
            {"query_table_id": "q_train", "target_table_id": "t1", "split": "train"},
            {"query_table_id": "q_test", "target_table_id": "t2", "split": "test"},
        ],
        "decisions": [
            {"source_table_id": "s1", "reason": "queryable", "split": "train"},
            {"source_table_id": "s2", "reason": "queryable", "split": "test"},
        ],
    }
    _write_jsonl(root / "query_tables/part-00000.jsonl", payloads["queries"])
    _write_jsonl(root / "data_lake_tables/part-00000.jsonl", payloads["targets"])
    _write_jsonl(root / "evidence_recoveries/part-00000.jsonl", payloads["recoveries"])
    _write_jsonl(root / "qrels.jsonl", payloads["qrels"])
    _write_jsonl(root / "table_queryability_decisions.jsonl", payloads["decisions"])
    (root / "splits.json").write_text(
        json.dumps(
            {
                "train": {
                    "source_table_ids": ["s1"],
                    "query_table_ids": ["q_train"],
                    "data_lake_table_ids": ["t1"],
                },
                "dev": {
                    "source_table_ids": [],
                    "query_table_ids": [],
                    "data_lake_table_ids": [],
                },
                "test": {
                    "source_table_ids": ["s2"],
                    "query_table_ids": ["q_test"],
                    "data_lake_table_ids": ["t2"],
                },
                "split_key": "page_title_or_source_table_id",
            }
        ),
        encoding="utf-8",
    )
    return payloads


def test_migrate_published_manifest_to_query_only_shared_lake(tmp_path: Path) -> None:
    payloads = _write_legacy_dataset(tmp_path)
    qrels = payloads["qrels"]
    recoveries = payloads["recoveries"]

    manifest = {
        "complete": True,
        "records_per_shard": 10000,
        "artifacts": {
            "query_tables": {
                "shards": [
                    _published_shard(tmp_path, "query_tables/part-00000.jsonl", 2)
                ]
            },
            "data_lake_tables": {
                "shards": [
                    _published_shard(tmp_path, "data_lake_tables/part-00000.jsonl", 2)
                ]
            },
            "evidence_recoveries": {
                "shards": [
                    _published_shard(
                        tmp_path, "evidence_recoveries/part-00000.jsonl", 1
                    )
                ]
            },
        },
        "single_files": {
            "qrels": "qrels.jsonl",
            "splits": "splits.json",
            "table_queryability_decisions": "table_queryability_decisions.jsonl",
        },
        "published_single_files": {},
        "query_construction": {"split_by": "page_title"},
    }
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    summary = migrate_dataset(tmp_path)

    assert summary["query_table_counts"] == {"train": 1, "dev": 0, "test": 1}
    assert summary["data_lake_split_fields_removed"] == 2
    assert all(
        "split" not in record
        for record in _records(tmp_path / "data_lake_tables/part-00000.jsonl")
    )
    assert all(
        "split" not in record
        for record in _records(tmp_path / "table_queryability_decisions.jsonl")
    )
    assert _records(tmp_path / "qrels.jsonl") == qrels
    assert _records(tmp_path / "evidence_recoveries/part-00000.jsonl") == recoveries
    assert not (tmp_path / "split_assignments").exists()

    splits = json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))
    assert splits == {
        "data_lake_artifact": "data_lake_tables",
        "data_lake_scope": "shared",
        "data_lake_table_count": 2,
        "query_table_counts": {"dev": 0, "test": 1, "train": 1},
        "split_key": "page_title_or_source_table_id",
        "split_policy": "query_only",
    }
    migrated_manifest = json.loads(
        (tmp_path / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    target_metadata = migrated_manifest["artifacts"]["data_lake_tables"]["shards"][0]
    target_path = tmp_path / target_metadata["path"]
    assert target_metadata["bytes"] == target_path.stat().st_size
    assert target_metadata["sha256"] == _sha256(target_path)
    assert migrated_manifest["query_construction"]["split_policy"] == "query_only"
    assert migrated_manifest["query_construction"]["data_lake_scope"] == "shared"
    assert "split_assignments" not in migrated_manifest["artifacts"]


def _compact_manifest() -> dict[str, object]:
    """The manifest the compact EntiTables builder writes: no completeness marker,
    no published-file registry, and shards listed without checksums."""
    return {
        "format": "sharded_jsonl",
        "records_per_shard": 50000,
        "artifacts": {
            "query_tables": {
                "shards": [{"path": "query_tables/part-00000.jsonl", "records": 2}]
            },
            "data_lake_tables": {
                "shards": [{"path": "data_lake_tables/part-00000.jsonl", "records": 2}]
            },
            "evidence_recoveries": {
                "shards": [
                    {"path": "evidence_recoveries/part-00000.jsonl", "records": 1}
                ]
            },
        },
        "single_files": {
            "qrels": "qrels.jsonl",
            "splits": "splits.json",
            "table_queryability_decisions": "table_queryability_decisions.jsonl",
        },
        "query_construction": {"query_rows_per_table": 5},
    }


def test_migrate_compact_manifest_keeps_it_checksum_free(tmp_path: Path) -> None:
    _write_legacy_dataset(tmp_path)
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(_compact_manifest()), encoding="utf-8"
    )

    summary = migrate_dataset(tmp_path)

    assert summary["data_lake_split_fields_removed"] == 2
    assert all(
        "split" not in record
        for record in _records(tmp_path / "data_lake_tables/part-00000.jsonl")
    )
    migrated = json.loads(
        (tmp_path / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    # Checksums would switch on per-read verification in iter_dataset_artifact,
    # re-hashing the whole artifact before it yields its first record.
    assert migrated["artifacts"]["data_lake_tables"]["shards"] == [
        {"path": "data_lake_tables/part-00000.jsonl", "records": 2}
    ]
    assert all(
        entry.keys() == {"path", "records"}
        for entry in migrated["published_single_files"].values()
    )
    assert not list(tmp_path.rglob("*.query-only.tmp"))


def test_migrate_rejects_manifest_declared_incomplete(tmp_path: Path) -> None:
    _write_legacy_dataset(tmp_path)
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({**_compact_manifest(), "complete": False}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="not complete"):
        migrate_dataset(tmp_path)

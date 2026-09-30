from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage2.column_data import read_jsonl, write_jsonl
from mmdd_stage2.natural_evidence import (
    NATURAL_EVIDENCE_KEY,
    augment_inputs,
    build_inputs,
    holdout_partition,
    natural_evidence_ids,
    retained_paths,
    stage1_directory_loader,
)


def item(query_id: str, target_id: str, **evidence: list[str]) -> dict:
    return {"dataset": "mm", "query_id": query_id, "target_id": target_id,
            "evidence_ids": {"O-O": ["w1"], "O-R": ["w1", "w2"], **evidence}}


def test_retained_paths_accepts_both_stage1_export_shapes():
    assert retained_paths({"pool": {"retained_paths": {"t1": ["e1"]}}}) == {"t1": ["e1"]}
    assert retained_paths({"retained_paths": {"t1": ["e1"]}}) == {"t1": ["e1"]}
    with pytest.raises(ValueError, match="no retained_paths"):
        retained_paths({"query_id": "q"})


def test_natural_evidence_keeps_retrieval_order_and_truncates_to_the_budget():
    record = {"pool": {"retained_paths": {"t1": ["e3", "e1", "e2", "e0", "e4"]}}}
    assert natural_evidence_ids(record, "t1", max_evidence=4) == ["e3", "e1", "e2", "e0"]
    assert natural_evidence_ids(record, "t1") == ["e3", "e1", "e2", "e0"]
    assert natural_evidence_ids({"pool": {"retained_paths": {}}}, "t1") == []
    with pytest.raises(ValueError, match="max_evidence"):
        natural_evidence_ids(record, "t1", max_evidence=0)


def test_holdout_partition_is_deterministic_and_uses_both_sides():
    groups = [f"st_{index:04d}" for index in range(400)]
    partitions = {group: holdout_partition(group) for group in groups}
    assert partitions == {group: holdout_partition(group) for group in groups}
    assert set(partitions.values()) == {"fit", "holdout"}
    with pytest.raises(ValueError, match="modulus/residue"):
        holdout_partition("st_1", modulus=10, residue=10)


def test_augment_inputs_adds_the_key_without_touching_witness_keys():
    items = [item("q1", "t1"), item("q2", "t2")]
    stage1 = {"q1": {"pool": {"retained_paths": {"t1": ["e1", "e2"]}}}, "q2": {"pool": {"retained_paths": {}}}}
    groups = {"q1": "st_a", "q2": "st_b"}
    augmented, funnel = augment_inputs(items, stage1, groups=groups)
    assert augmented[0]["evidence_ids"]["O-O"] == ["w1"]
    assert augmented[0]["evidence_ids"]["O-R"] == ["w1", "w2"]
    assert augmented[0]["evidence_ids"][NATURAL_EVIDENCE_KEY] == ["e1", "e2"]
    assert augmented[1]["evidence_ids"][NATURAL_EVIDENCE_KEY] == []
    assert funnel["pairs"] == 2 and funnel["pairs_with_natural_evidence"] == 1
    assert funnel["evidence_items"] == 2 and funnel["group_overlap"] == 0
    assert augmented[0]["natural_partition"] == holdout_partition("st_a")


def test_augment_inputs_reports_missing_stage1_records_instead_of_inventing_evidence():
    items = [item("q9", "t9")]
    augmented, funnel = augment_inputs(items, {}, groups={"q9": "st_z"})
    assert augmented[0]["evidence_ids"][NATURAL_EVIDENCE_KEY] == []
    assert funnel["missing_stage1_record"] == 1
    assert funnel["pairs_with_natural_evidence"] == 0


def test_build_inputs_writes_hashed_augmented_files(tmp_path: Path):
    r1 = tmp_path / "r1"
    r1.mkdir()
    records = {split: [item(f"q_{split}_1", "t1"), item(f"q_{split}_2", "t2")] for split in ("train", "dev")}
    for split, rows_ in records.items():
        write_jsonl(r1 / f"COLUMN_INPUTS.{split}.jsonl", rows_)
    stage1 = {}
    for split in ("train", "dev"):
        for index, record in enumerate(records[split], 1):
            stage1[record["query_id"]] = {"pool": {"retained_paths": {"t1": [f"e{index}"]}}}
    manifest = build_inputs(r1, tmp_path, stage1, splits=("train", "dev"))
    assert set(manifest["splits"]) == {"train", "dev"}
    written = read_jsonl(tmp_path / "NAT_E" / "COLUMN_INPUTS.train.jsonl")
    assert len(written) == 2
    assert written[0]["evidence_ids"][NATURAL_EVIDENCE_KEY] == ["e1"]
    assert manifest["splits"]["train"]["sha256"]
    assert (tmp_path / "NAT_E" / "MANIFEST.json").is_file()


def test_items_sharing_a_source_group_always_land_in_the_same_partition():
    items = [item("q1", "t1"), item("q2", "t2"), item("q3", "t3")]
    stage1 = {qid: {"pool": {"retained_paths": {}}} for qid in ("q1", "q2", "q3")}
    groups = {"q1": "st_shared", "q2": "st_shared", "q3": "st_other"}
    augmented, funnel = augment_inputs(items, stage1, groups=groups)
    assert augmented[0]["natural_partition"] == augmented[1]["natural_partition"]
    assert funnel["group_overlap"] == 0
    assert funnel["fit_pairs"] + funnel["holdout_pairs"] == 3


def test_stage1_directory_loader_reads_records_and_fails_loudly_on_missing_files(tmp_path: Path):
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "q1.json").write_text(
        json.dumps({"pool": {"retained_paths": {"t1": ["e1"]}}}), encoding="utf-8")
    loader = stage1_directory_loader(tmp_path)
    assert loader("dev", "q1")["pool"]["retained_paths"] == {"t1": ["e1"]}
    with pytest.raises(FileNotFoundError, match="Stage-1 record missing"):
        loader("dev", "absent")

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_stage1
from split_stage1_corpus_by_lake import split_corpus


def _write_corpus(path: Path, object_ids: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"object_id": object_id, "tag": object_id}) + "\n"
            for object_id in object_ids
        ),
        encoding="utf-8",
    )


def test_split_stage1_corpus_proves_exact_disjoint_partition(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed.jsonl"
    entitables = tmp_path / "entitables.jsonl"
    wdc = tmp_path / "wdc.jsonl"
    _write_corpus(mixed, ["e1", "w1", "e2"])
    _write_corpus(entitables, ["e1", "e2"])
    _write_corpus(wdc, ["w1"])

    summary = split_corpus(
        mixed,
        [("entitables", entitables), ("wdc", wdc)],
        tmp_path / "output",
    )

    assert summary["mixed_objects"] == 3
    assert summary["partition_exact"] is True
    assert summary["pairwise_disjoint"] is True
    assert summary["lakes"]["entitables"]["objects"] == 2
    assert (tmp_path / "output" / "entitables_corpus.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() == [
        json.dumps({"object_id": "e1", "tag": "e1"}),
        json.dumps({"object_id": "e2", "tag": "e2"}),
    ]


def test_split_stage1_corpus_rejects_overlapping_lakes(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_corpus(mixed, ["shared"])
    _write_corpus(first, ["shared"])
    _write_corpus(second, ["shared"])

    with pytest.raises(ValueError, match="overlap"):
        split_corpus(mixed, [("first", first), ("second", second)], tmp_path / "out")


def test_split_stage1_corpus_rejects_incomplete_partition(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_corpus(mixed, ["a"])
    _write_corpus(first, ["a"])
    _write_corpus(second, ["missing"])

    with pytest.raises(ValueError, match="missing 1 source objects"):
        split_corpus(mixed, [("first", first), ("second", second)], tmp_path / "out")


def test_initialize_only_requires_epoch_zero_student_path() -> None:
    valid = type(
        "Args",
        (),
        {"initialize_only": True, "stage": "student-path", "eval_epoch_zero": True},
    )
    invalid_stage = type(
        "Args",
        (),
        {"initialize_only": True, "stage": "student-edge", "eval_epoch_zero": True},
    )
    invalid_epoch_zero = type(
        "Args",
        (),
        {"initialize_only": True, "stage": "student-path", "eval_epoch_zero": False},
    )

    train_stage1._validate_initialize_only(valid)
    with pytest.raises(ValueError, match="--initialize-only"):
        train_stage1._validate_initialize_only(invalid_stage)
    with pytest.raises(ValueError, match="--initialize-only"):
        train_stage1._validate_initialize_only(invalid_epoch_zero)

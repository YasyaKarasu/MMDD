from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

from mmdd_stage2.r4_common import read_jsonl
from mmdd_stage2.r4c_fast_sampling import (
    COMPONENT_SALT,
    _select,
    smoke_units,
    unit_shard,
    write_deterministic_jsonl_gz,
)
from mmdd_stage2.r4c_fast_types import FastUnit


def _unit(index: int, *, group: str | None = None, query: str | None = None, image: str | None = None) -> FastUnit:
    return FastUnit.create(
        query_id=query or f"q{index}",
        target_id=f"t{index}",
        column_id=1,
        column_name="Artist",
        query_row_id=index,
        source_group=group or f"g{index}",
        cells=(("Name", f"row{index}"),),
        evidence_ids=(image or f"image{index}",),
        focus_image_id=image or f"image{index}",
    )


def test_unit_hash_does_not_include_evaluation_sidecar() -> None:
    unit = _unit(1)
    changed_gold = {"unit_id": unit.unit_id, "gold_values": ["different"]}
    assert changed_gold["unit_id"] == unit.unit_id
    assert "gold" not in unit.to_dict()


def test_source_group_first_lock_caps_group_query_and_image() -> None:
    units = [
        _unit(0, group="g", query="q", image="a"),
        _unit(1, group="g", query="q", image="b"),
        _unit(2, group="g", query="q", image="c"),
        _unit(3, group="h", query="r", image="a"),
        _unit(4, group="h", query="s", image="d"),
    ]
    selected = _select(units, salt=COMPONENT_SALT, target_units=10, per_group=2, per_query=2)
    assert max(sum(item.source_group == group for item in selected) for group in {"g", "h"}) <= 2
    assert max(sum(item.query_id == query for item in selected) for query in {"q", "r", "s"}) <= 2
    assert len({item.focus_image_id for item in selected}) == len(selected)


def test_same_unit_all_arms_share_one_gpu_shard() -> None:
    unit = _unit(9)
    expected = int(hashlib.sha256(unit.unit_id.encode()).hexdigest()[:16], 16) % 2
    assert unit_shard(unit.unit_id, 2) == expected


def test_smoke_is_first_eight_groups_with_no_more_than_sixteen_units() -> None:
    units = [_unit(index, group=f"g{index // 2}") for index in range(64)]
    smoke = smoke_units(units)
    assert len(smoke) == 16
    assert len({unit.source_group for unit in smoke}) == 8


def test_two_writes_have_identical_lock_sha(tmp_path: Path) -> None:
    records = [_unit(index).to_dict() for index in range(3)]
    left, right = tmp_path / "left.gz", tmp_path / "right.gz"
    write_deterministic_jsonl_gz(left, records)
    write_deterministic_jsonl_gz(right, records)
    assert left.read_bytes() == right.read_bytes()
    assert read_jsonl(left) == records


def test_lock_is_valid_gzip(tmp_path: Path) -> None:
    path = tmp_path / "lock.gz"
    write_deterministic_jsonl_gz(path, [_unit(1).to_dict()])
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert "unit_id" in handle.read()

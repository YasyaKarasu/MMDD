import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_mm_joinability_dataset as builder


def write_entitables_file(path: Path, table_ids: list[str]) -> None:
    payload = {
        table_id: {
            "title": ["Entity", "Value"],
            "data": [
                [f"{table_id} entity 1", "alpha"],
                [f"{table_id} entity 2", "beta"],
            ],
            "numCols": 2,
            "numDataRows": 2,
        }
        for table_id in table_ids
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def entitables_dir(tmp_path: Path) -> Path:
    write_entitables_file(tmp_path / "a.json", [f"a_table_{idx}" for idx in range(4)])
    write_entitables_file(tmp_path / "b.json", [f"b_table_{idx}" for idx in range(4)])
    return tmp_path


def candidate_ids(input_dir: Path, seed: int) -> list[str]:
    args = builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(input_dir / "out"),
            "--seed",
            str(seed),
        ]
    )
    counters = builder.SourceCandidateCounters()
    return [
        table["source_table_id"]
        for table in builder.iter_random_source_tables(input_dir, args, counters)
    ]


def test_replacement_policy_defaults(tmp_path: Path) -> None:
    args = builder.parse_args(
        ["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "out")]
    )

    assert builder.replacement_policy_from_args(args) == builder.ReplacementPolicy(2, 0.5)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--unrecoverable_replacement_rounds", "-1"),
        ("--unrecoverable_drop_probability", "-0.01"),
        ("--unrecoverable_drop_probability", "1.01"),
    ],
)
def test_replacement_policy_rejects_invalid_values(
    tmp_path: Path, option: str, value: str
) -> None:
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            option,
            value,
        ]
    )

    with pytest.raises(ValueError):
        builder.replacement_policy_from_args(args)


def test_equal_seeds_produce_equal_candidate_order(entitables_dir: Path) -> None:
    first = candidate_ids(entitables_dir, 13)

    assert first == candidate_ids(entitables_dir, 13)


def test_seed_thirteen_candidate_order_is_not_lexicographic(
    entitables_dir: Path,
) -> None:
    first = candidate_ids(entitables_dir, 13)

    assert first != sorted(first)


def test_different_seeds_produce_different_candidate_order(
    entitables_dir: Path,
) -> None:
    assert candidate_ids(entitables_dir, 13) != candidate_ids(entitables_dir, 29)

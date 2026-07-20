import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_mm_joinability_dataset as builder


class StubRandom:
    def __init__(self, draws: list[float]):
        self.draws = iter(draws)

    def random(self) -> float:
        return next(self.draws)


def replacement_tables() -> list[dict[str, str]]:
    return [{"source_table_id": f"t{index}"} for index in range(5)]


def evaluator_for(
    queryability: dict[str, bool],
):
    def evaluate_batch(
        tables: list[dict[str, str]],
    ) -> list[builder.CandidateEvaluation]:
        return [
            builder.CandidateEvaluation(
                source_table=table,
                queryable=queryability[table["source_table_id"]],
                decision={"source_table_id": table["source_table_id"]},
            )
            for table in tables
        ]

    return evaluate_batch


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


def test_queryable_table_never_draws_or_replaces() -> None:
    tables = replacement_tables()
    discarded: list[str] = []

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(tables),
        target_count=1,
        policy=builder.ReplacementPolicy(rounds=2, drop_probability=0.5),
        rng=StubRandom([]),
        evaluate_batch=evaluator_for({"t0": True}),
        discard_table=discarded.append,
    )

    assert [item.source_table["source_table_id"] for item in selection.final_evaluations] == [
        "t0"
    ]
    assert selection.rounds == [builder.ReplacementRoundStats(0, 1, 0, 0, 0, 0)]
    assert selection.candidates_consumed == 1
    assert selection.candidate_exhausted is False
    assert selection.unfilled_slots == 0
    assert discarded == []


def test_failed_table_is_retained_when_draw_equals_probability() -> None:
    tables = replacement_tables()
    discarded: list[str] = []

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(tables),
        target_count=1,
        policy=builder.ReplacementPolicy(rounds=2, drop_probability=0.5),
        rng=StubRandom([0.5]),
        evaluate_batch=evaluator_for({"t0": False}),
        discard_table=discarded.append,
    )

    assert [item.source_table["source_table_id"] for item in selection.final_evaluations] == [
        "t0"
    ]
    assert selection.rounds == [builder.ReplacementRoundStats(0, 1, 1, 0, 1, 0)]
    assert selection.candidates_consumed == 1
    assert discarded == []


def test_failed_slot_replaces_twice_then_retains_at_limit() -> None:
    tables = replacement_tables()
    discarded: list[str] = []

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(tables),
        target_count=2,
        policy=builder.ReplacementPolicy(rounds=2, drop_probability=0.5),
        rng=StubRandom([0.1, 0.1]),
        evaluate_batch=evaluator_for(
            {"t0": False, "t1": True, "t2": False, "t3": False}
        ),
        discard_table=discarded.append,
    )

    assert [item.source_table["source_table_id"] for item in selection.final_evaluations] == [
        "t3",
        "t1",
    ]
    assert selection.rounds == [
        builder.ReplacementRoundStats(0, 2, 1, 1, 0, 1),
        builder.ReplacementRoundStats(1, 1, 1, 1, 0, 1),
        builder.ReplacementRoundStats(2, 1, 1, 0, 1, 0),
    ]
    assert selection.candidates_consumed == 4
    assert selection.candidate_exhausted is False
    assert selection.unfilled_slots == 0
    assert discarded == ["t0", "t2"]


def test_exhausted_replacement_retains_failed_table_without_cleanup() -> None:
    tables = replacement_tables()
    discarded: list[str] = []

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(tables[:1]),
        target_count=1,
        policy=builder.ReplacementPolicy(rounds=2, drop_probability=1.0),
        rng=StubRandom([0.0]),
        evaluate_batch=evaluator_for({"t0": False}),
        discard_table=discarded.append,
    )

    assert [item.source_table["source_table_id"] for item in selection.final_evaluations] == [
        "t0"
    ]
    assert selection.rounds == [builder.ReplacementRoundStats(0, 1, 1, 0, 1, 0)]
    assert selection.candidates_consumed == 1
    assert selection.candidate_exhausted is True
    assert selection.unfilled_slots == 0
    assert discarded == []

import gc
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_mm_joinability_dataset as builder
import build_mm_table_dataset as table_builder


class StubRandom:
    def __init__(self, draws: list[float]):
        self.draws = iter(draws)

    def random(self) -> float:
        return next(self.draws)


class TqdmSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, iterable=None, **kwargs: object):
        call = {
            "iterable": iterable,
            "postfixes": [],
            "updates": [],
            "closed": False,
            **kwargs,
        }
        self.calls.append(call)

        class SpyBar:
            def __iter__(self):
                return iter(iterable)

            def set_postfix(self, **postfix: object) -> None:
                call["postfixes"].append(postfix)

            def update(self, amount: int = 1) -> None:
                call["updates"].append(amount)

            def close(self) -> None:
                call["closed"] = True

        return SpyBar()


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
                [f"[{table_id}_entity_1|{table_id} entity 1]", "alpha"],
                [f"[{table_id}_entity_2|{table_id} entity 2]", "beta"],
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


def candidate_ids(
    input_dir: Path,
    seed: int,
    *,
    max_source_tables: int | None = None,
    replacement_rounds: int = 2,
) -> list[str]:
    extra_args: list[str] = []
    if max_source_tables is not None:
        extra_args.extend(["--max_source_tables", str(max_source_tables)])
    args = builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(input_dir / "out"),
            "--seed",
            str(seed),
            "--unrecoverable_replacement_rounds",
            str(replacement_rounds),
            *extra_args,
        ]
    )
    counters = builder.SourceCandidateCounters()
    return [
        table["source_table_id"]
        for table in builder.iter_random_source_tables(input_dir, args, counters)
    ]


def expected_global_candidate_ids(
    input_dir: Path,
    seed: int,
    *,
    limit: int,
) -> list[str]:
    priorities = []
    for json_file in sorted(input_dir.glob("*.json")):
        relative_path = json_file.relative_to(input_dir).as_posix()
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        for table_id in payload:
            priority = int(
                builder.stable_hash(
                    "global-source-table", seed, relative_path, table_id, length=40
                ),
                16,
            )
            priorities.append((priority, relative_path, table_id))
    return [
        f"st_{table_id}_{builder.stable_hash(relative_path, table_id, length=10)}"
        for _priority, relative_path, table_id in sorted(priorities)[:limit]
    ]


def exercise_preparation_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    no_model_progress: bool,
) -> TqdmSpy:
    spy = TqdmSpy()
    monkeypatch.setattr(builder, "tqdm", spy)
    args_list = [
        "--input_dir",
        str(tmp_path),
        "--output_dir",
        str(tmp_path / "out"),
    ]
    if no_model_progress:
        args_list.append("--no_model_progress")
    args = builder.parse_args(args_list)

    write_entitables_file(tmp_path / "a.json", ["a_table"])
    write_entitables_file(tmp_path / "b.json", ["b_table"])
    list(builder.iter_random_source_tables(tmp_path, args, builder.SourceCandidateCounters()))

    source_tables = [
        {"source_table_id": "first", "entity_ids": {"e1"}},
        {"source_table_id": "second", "entity_ids": {"e2", "e3"}},
    ]
    context = SimpleNamespace(
        eligible_entity_ids=set(),
        entity_records={},
        max_entities=None,
        entity_to_assets={"e1": [], "e2": [], "e3": []},
        wiki_to_entity_id={},
    )
    monkeypatch.setattr(builder, "update_entities_from_table", lambda *_args: None)
    monkeypatch.setattr(
        builder,
        "_candidate_entity_ids",
        lambda table, _wiki_to_entity_id: table["entity_ids"],
    )
    builder.prepare_candidate_batch(source_tables, context, args)
    return spy


def test_preparation_progress_reports_totals_and_eligible_entities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = exercise_preparation_progress(
        tmp_path, monkeypatch, no_model_progress=False
    )

    assert [
        {
            "desc": call["desc"],
            "total": call["total"],
            "unit": call["unit"],
            "disable": call["disable"],
        }
        for call in spy.calls
    ] == [
        {
            "desc": "Scanning EntiTables for global sample",
            "total": 2,
            "unit": "file",
            "disable": False,
        },
        {
            "desc": "Materializing global EntiTables sample",
            "total": 1,
            "unit": "chunk",
            "disable": False,
        },
        {
            "desc": "Preparing candidate batch materials",
            "total": 2,
            "unit": "table",
            "disable": False,
        },
    ]
    assert spy.calls[2]["postfixes"] == [
        {"eligible_entities": 1},
        {"eligible_entities": 3},
    ]


def test_no_model_progress_disables_preparation_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = exercise_preparation_progress(tmp_path, monkeypatch, no_model_progress=True)

    assert [call["disable"] for call in spy.calls] == [True, True, True]


def test_no_model_progress_suppresses_batch_helper_internal_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--no_model_progress",
        ]
    )
    monkeypatch.setattr(builder, "tqdm", None)
    monkeypatch.setattr(
        table_builder,
        "tqdm",
        lambda *_args, **_kwargs: pytest.fail(
            "batch helper emitted progress with --no_model_progress"
        ),
    )

    class FakeWikipediaClient:
        api_failures = 0

        def __init__(self) -> None:
            self.page_cache: dict[str, dict[str, object]] = {}
            self.image_cache: dict[str, dict[str, object]] = {}

        def get_pages(self, wiki_titles: object) -> dict[str, dict[str, object]]:
            pages = {
                str(title): {
                    "wiki_title": str(title),
                    "images": [{"title": f"File:{title}.jpg"}],
                }
                for title in wiki_titles
            }
            self.page_cache.update(pages)
            return pages

        def get_imageinfos(self, file_titles: object) -> dict[str, dict[str, object]]:
            return {
                str(title): {
                    "file_title": str(title),
                    "url": f"https://images.example/{title}",
                }
                for title in file_titles
            }

        def download_image(
            self, _imageinfo: object, asset_id: str
        ) -> dict[str, object]:
            return {
                "local_path": str(tmp_path / f"{asset_id}.jpg"),
                "relative_path": f"{asset_id}.jpg",
                "file_name": f"{asset_id}.jpg",
                "bytes": 1,
                "sha256": "sha",
                "downloaded": True,
            }

    wikipedia = FakeWikipediaClient()
    cache = builder.ExtractionCache(tmp_path / "model.jsonl")
    assets: dict[str, dict[str, object]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    registry = builder.CandidateMaterialRegistry(
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extraction_cache=cache,
    )
    context = builder.CandidateEvaluationContext(
        entity_records={},
        wiki_to_entity_id={},
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extractor=None,
        cache=cache,
        progress=None,
        concurrency_state=SimpleNamespace(),
        registry=registry,
    )
    tables = [
        {
            "source_table_id": "batch",
            "rows": [
                {
                    "row_id": 0,
                    "cells": [
                        {
                            "column_index": index,
                            "column_name": "Entity",
                            "text": title,
                            "wiki_title": title,
                        }
                        for index, title in enumerate(("Page A", "Page B"))
                    ],
                }
            ],
        }
    ]

    builder.prepare_candidate_batch(tables, context, args)

    assert len(assets) == 2


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


def test_equal_seeds_ignore_source_file_enumeration_order(
    entitables_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovered = [entitables_dir / "a.json", entitables_dir / "b.json"]
    enumeration_orders = iter((discovered, list(reversed(discovered))))

    monkeypatch.setattr(
        Path,
        "rglob",
        lambda _path, _pattern: iter(next(enumeration_orders)),
    )

    assert candidate_ids(entitables_dir, 13) == candidate_ids(entitables_dir, 13)


def test_seed_thirteen_candidate_order_is_not_lexicographic(
    entitables_dir: Path,
) -> None:
    first = candidate_ids(entitables_dir, 13)

    assert first != sorted(first)


def test_different_seeds_produce_different_candidate_order(
    entitables_dir: Path,
) -> None:
    assert candidate_ids(entitables_dir, 13) != candidate_ids(entitables_dir, 29)


def test_global_sample_filters_tables_without_candidate_entity_column(
    tmp_path: Path,
) -> None:
    payload = {
        "without_entity": {
            "title": ["Name", "Value"],
            "data": [["Alpha", "one"], ["Beta", "two"]],
            "numCols": 2,
            "numDataRows": 2,
        },
        "with_entity": {
            "title": ["Entity", "Value"],
            "data": [
                ["[Alpha|Alpha]", "one"],
                ["[Beta|Beta]", "two"],
            ],
            "numCols": 2,
            "numDataRows": 2,
        },
    }
    (tmp_path / "tables.json").write_text(json.dumps(payload), encoding="utf-8")
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--max_source_tables",
            "2",
            "--unrecoverable_replacement_rounds",
            "0",
        ]
    )
    counters = builder.SourceCandidateCounters()

    selected = list(builder.iter_random_source_tables(tmp_path, args, counters))

    assert [table["source_table_id"] for table in selected] == [
        f"st_with_entity_{builder.stable_hash('tables.json', 'with_entity', length=10)}"
    ]
    assert counters.skipped_tables == 1
    assert counters.skip_reasons == {"no_candidate_entity_column": 1}


def test_global_capped_sample_scans_every_file_before_first_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for filename in ("a.json", "b.json", "z.json"):
        write_entitables_file(tmp_path / filename, [f"{filename[0]}_table"])
    reads: list[str] = []
    real_read = builder.read_entitables_json

    def recording_read(path: Path):
        reads.append(path.name)
        return real_read(path)

    monkeypatch.setattr(builder, "read_entitables_json", recording_read)
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--max_source_tables",
            "1",
            "--unrecoverable_replacement_rounds",
            "0",
        ]
    )

    first = next(
        builder.iter_random_source_tables(
            tmp_path, args, builder.SourceCandidateCounters()
        )
    )

    assert reads[:3] == ["a.json", "b.json", "z.json"]
    assert first["source_table_id"].startswith(("st_a_table_", "st_b_table_", "st_z_table_"))


def test_global_sample_uses_stable_priority_across_all_files(tmp_path: Path) -> None:
    write_entitables_file(tmp_path / "a.json", [f"a_table_{idx}" for idx in range(5)])
    write_entitables_file(tmp_path / "z.json", [f"z_table_{idx}" for idx in range(5)])

    actual = candidate_ids(
        tmp_path, 41, max_source_tables=2, replacement_rounds=0
    )

    assert actual == expected_global_candidate_ids(tmp_path, 41, limit=2)
    assert any(table_id.startswith("st_z_") for table_id in actual)


def test_global_sample_capacity_covers_all_replacement_rounds(tmp_path: Path) -> None:
    write_entitables_file(tmp_path / "a.json", [f"a_table_{idx}" for idx in range(4)])
    write_entitables_file(tmp_path / "z.json", [f"z_table_{idx}" for idx in range(4)])

    actual = candidate_ids(
        tmp_path, 17, max_source_tables=2, replacement_rounds=2
    )

    assert len(actual) == 6
    assert actual == expected_global_candidate_ids(tmp_path, 17, limit=6)


def test_global_sample_is_reproducible_seeded_and_enumeration_invariant(
    entitables_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovered = [entitables_dir / "a.json", entitables_dir / "b.json"]
    enumeration_orders = iter(
        (discovered, list(reversed(discovered)), discovered, discovered)
    )
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda _path, _pattern: iter(next(enumeration_orders)),
    )

    first = candidate_ids(
        entitables_dir, 13, max_source_tables=2, replacement_rounds=1
    )
    reversed_enumeration = candidate_ids(
        entitables_dir, 13, max_source_tables=2, replacement_rounds=1
    )
    repeated = candidate_ids(
        entitables_dir, 13, max_source_tables=2, replacement_rounds=1
    )
    alternate_seed = candidate_ids(
        entitables_dir, 29, max_source_tables=2, replacement_rounds=1
    )

    assert first == reversed_enumeration == repeated
    assert first != alternate_seed


@pytest.mark.parametrize(
    "failure_mode",
    ["file_read", "missing_table", "parse_failure"],
)
def test_global_rematerialization_fails_fast_without_changing_scan_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    json_file = tmp_path / "a.json"
    write_entitables_file(json_file, ["a_table"])
    real_read = builder.read_entitables_json
    real_parse = builder.parse_source_table
    read_count = 0
    parse_count = 0

    def sabotaged_read(path: Path):
        nonlocal read_count
        read_count += 1
        if read_count == 2:
            if failure_mode == "file_read":
                return None
            if failure_mode == "missing_table":
                return {}
        return real_read(path)

    def sabotaged_parse(*args, **kwargs):
        nonlocal parse_count
        parse_count += 1
        if failure_mode == "parse_failure" and parse_count == 2:
            return SimpleNamespace(source_table=None, skip_reason="too_few_rows")
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(builder, "read_entitables_json", sabotaged_read)
    monkeypatch.setattr(builder, "parse_source_table", sabotaged_parse)
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--max_source_tables",
            "1",
            "--unrecoverable_replacement_rounds",
            "0",
        ]
    )
    counters = builder.SourceCandidateCounters()

    with pytest.raises(RuntimeError, match=r"a\.json.*a_table"):
        list(builder.iter_random_source_tables(tmp_path, args, counters))

    assert counters.processed_tables == 1
    assert counters.skipped_tables == 0
    assert counters.skip_reasons == {}


def test_global_materialization_progress_updates_before_yield_and_closes_on_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_entitables_file(tmp_path / "a.json", [f"table_{idx}" for idx in range(4)])
    spy = TqdmSpy()
    monkeypatch.setattr(builder, "tqdm", spy)
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--max_source_tables",
            "2",
            "--unrecoverable_replacement_rounds",
            "1",
        ]
    )
    candidates = builder.iter_random_source_tables(
        tmp_path, args, builder.SourceCandidateCounters()
    )

    initial_chunk = [next(candidates), next(candidates)]

    assert len(initial_chunk) == 2
    materialization_call = spy.calls[1]
    assert materialization_call["updates"] == [1]
    assert materialization_call["closed"] is False

    candidates.close()

    assert materialization_call["closed"] is True


def test_queryable_table_never_draws_or_replaces() -> None:
    tables = replacement_tables()
    discarded: list[str] = []

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(tables),
        target_count=1,
        policy=builder.ReplacementPolicy(rounds=2, drop_probability=0.5),
        rng=StubRandom([]),
        evaluate_batch=evaluator_for({"t0": True}),
        discard_tables=lambda table_ids: discarded.extend(table_ids),
    )

    assert [item.source_table["source_table_id"] for item in selection.final_evaluations] == [
        "t0"
    ]
    assert selection.rounds == [builder.ReplacementRoundStats(0, 1, 0, 0, 0, 0)]
    assert selection.candidates_consumed == 1
    assert selection.candidate_exhausted is False
    assert selection.unfilled_slots == 0
    assert discarded == []


def test_failed_table_not_selected_in_one_pass_remains_eligible_later() -> None:
    tables = replacement_tables()
    discarded: list[str] = []

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(tables),
        target_count=1,
        policy=builder.ReplacementPolicy(rounds=2, drop_probability=0.5),
        rng=StubRandom([0.5, 0.1]),
        evaluate_batch=evaluator_for({"t0": False, "t1": True}),
        discard_tables=lambda table_ids: discarded.extend(table_ids),
    )

    assert [item.source_table["source_table_id"] for item in selection.final_evaluations] == [
        "t1"
    ]
    assert selection.rounds == [
        builder.ReplacementRoundStats(0, 1, 1, 0, 1, 0),
        builder.ReplacementRoundStats(1, 0, 1, 1, 0, 1),
        builder.ReplacementRoundStats(2, 1, 0, 0, 0, 0),
    ]
    assert selection.candidates_consumed == 2
    assert discarded == ["t0"]


def test_each_pass_samples_from_complete_current_failed_pool() -> None:
    discarded: list[str] = []

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(replacement_tables()),
        target_count=2,
        policy=builder.ReplacementPolicy(rounds=2, drop_probability=0.5),
        rng=StubRandom([0.9, 0.1, 0.1, 0.9]),
        evaluate_batch=evaluator_for(
            {"t0": False, "t1": False, "t2": False, "t3": True}
        ),
        discard_tables=lambda table_ids: discarded.extend(table_ids),
    )

    assert [item.source_table["source_table_id"] for item in selection.final_evaluations] == [
        "t3",
        "t2",
    ]
    assert selection.rounds == [
        builder.ReplacementRoundStats(0, 2, 2, 1, 1, 1),
        builder.ReplacementRoundStats(1, 1, 2, 1, 1, 1),
        builder.ReplacementRoundStats(2, 1, 1, 0, 1, 0),
    ]
    assert selection.candidates_consumed == 4
    assert discarded == ["t1", "t0"]


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
        discard_tables=lambda table_ids: discarded.extend(table_ids),
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


def test_model_start_waits_for_initial_batch_preparation() -> None:
    tables = replacement_tables()[:3]
    events: list[str] = []

    def table_ids(batch: list[dict[str, str]]) -> str:
        return ",".join(table["source_table_id"] for table in batch)

    def start_after_preparation(initial_batch: list[dict[str, str]]) -> None:
        events.append(f"prepare:{table_ids(initial_batch)}")
        events.append("start")

    def evaluate_batch(
        batch: list[dict[str, str]],
    ) -> list[builder.CandidateEvaluation]:
        ids = table_ids(batch)
        if events[-1] != "start":
            events.append(f"prepare:{ids}")
        events.append(f"evaluate:{ids}")
        return [
            builder.CandidateEvaluation(
                source_table=table,
                queryable=table["source_table_id"] != "t0",
                decision={},
            )
            for table in batch
        ]

    builder.run_replacement_rounds(
        candidate_tables=iter(tables),
        target_count=2,
        policy=builder.ReplacementPolicy(rounds=1, drop_probability=1.0),
        rng=StubRandom([0.0]),
        evaluate_batch=evaluate_batch,
        discard_tables=lambda _table_ids: None,
        on_initial_batch=start_after_preparation,
    )

    assert events == [
        "prepare:t0,t1",
        "start",
        "evaluate:t0,t1",
        "prepare:t2",
        "evaluate:t2",
    ]
    assert events.count("start") == 1


def test_round_preparation_precedes_initial_and_replacement_evaluation() -> None:
    tables = replacement_tables()[:3]
    events: list[str] = []

    def table_ids(batch: list[dict[str, str]]) -> str:
        return ",".join(table["source_table_id"] for table in batch)

    def prepare_batch(batch: list[dict[str, str]]) -> None:
        events.append(f"prepare:{table_ids(batch)}")

    def start_models(batch: list[dict[str, str]]) -> None:
        events.append(f"start:{table_ids(batch)}")

    def evaluate_batch(
        batch: list[dict[str, str]],
    ) -> list[builder.CandidateEvaluation]:
        events.append(f"evaluate:{table_ids(batch)}")
        return [
            builder.CandidateEvaluation(
                source_table=table,
                queryable=table["source_table_id"] != "t0",
                decision={},
            )
            for table in batch
        ]

    builder.run_replacement_rounds(
        candidate_tables=iter(tables),
        target_count=2,
        policy=builder.ReplacementPolicy(rounds=1, drop_probability=1.0),
        rng=StubRandom([0.0]),
        prepare_batch=prepare_batch,
        evaluate_batch=evaluate_batch,
        discard_tables=lambda _table_ids: None,
        on_initial_batch_prepared=start_models,
    )

    assert events == [
        "prepare:t0,t1",
        "start:t0,t1",
        "evaluate:t0,t1",
        "prepare:t2",
        "evaluate:t2",
    ]


def test_replacement_round_flushes_discarded_tables_as_one_batch() -> None:
    discarded_batches: list[list[str]] = []

    builder.run_replacement_rounds(
        candidate_tables=iter(replacement_tables()[:4]),
        target_count=2,
        policy=builder.ReplacementPolicy(rounds=1, drop_probability=1.0),
        rng=StubRandom([0.0, 0.0]),
        evaluate_batch=evaluator_for(
            {"t0": False, "t1": False, "t2": True, "t3": True}
        ),
        discard_tables=lambda table_ids: discarded_batches.append(list(table_ids)),
    )

    assert discarded_batches == [["t0", "t1"]]


def test_exhausted_replacement_retains_failed_table_without_cleanup() -> None:
    tables = replacement_tables()
    discarded: list[str] = []

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(tables[:1]),
        target_count=1,
        policy=builder.ReplacementPolicy(rounds=2, drop_probability=1.0),
        rng=StubRandom([0.0]),
        evaluate_batch=evaluator_for({"t0": False}),
        discard_tables=lambda table_ids: discarded.extend(table_ids),
    )

    assert [item.source_table["source_table_id"] for item in selection.final_evaluations] == [
        "t0"
    ]
    assert selection.rounds == [builder.ReplacementRoundStats(0, 1, 1, 0, 1, 0)]
    assert selection.candidates_consumed == 1
    assert selection.candidate_exhausted is True
    assert selection.unfilled_slots == 0
    assert discarded == []


def test_replacement_registers_shared_dependencies_before_discard_cleanup(
    tmp_path: Path,
) -> None:
    assets: dict[str, dict[str, object]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    cache = builder.ExtractionCache(tmp_path / "model.jsonl")
    registry = builder.CandidateMaterialRegistry(
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=None,
        extraction_cache=cache,
    )
    shared = builder.CandidateDependencies(
        entities={"entity-shared"},
        assets={"asset-shared"},
        model_keys={"model-shared"},
    )
    work = {"fetches": 0, "inferences": 0}
    deleted_before_replacement_registration: list[bool] = []

    def evaluate_batch(
        tables: list[dict[str, str]],
    ) -> list[builder.CandidateEvaluation]:
        evaluations = []
        for table in tables:
            table_id = table["source_table_id"]
            if "asset-shared" not in assets:
                work["fetches"] += 1
                assets["asset-shared"] = {
                    "asset_id": "asset-shared",
                    "entity_id": "entity-shared",
                }
                entity_to_assets["entity-shared"] = ["asset-shared"]
            if "model-shared" not in cache.items:
                work["inferences"] += 1
                cache.items["model-shared"] = {"cache_key": "model-shared"}
            registry.register(table_id, shared)
            evaluations.append(
                builder.CandidateEvaluation(
                    source_table=table,
                    queryable=table_id == "t1",
                    decision={},
                )
            )
        return evaluations

    def discard_tables(table_ids: list[str]) -> None:
        deleted_before_replacement_registration.append(
            "t1" not in registry.dependencies
        )
        registry.discard_many(table_ids)

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(replacement_tables()[:2]),
        target_count=1,
        policy=builder.ReplacementPolicy(rounds=1, drop_probability=1.0),
        rng=StubRandom([0.0]),
        evaluate_batch=evaluate_batch,
        discard_tables=discard_tables,
    )

    assert [item.source_table["source_table_id"] for item in selection.final_evaluations] == [
        "t1"
    ]
    assert work == {"fetches": 1, "inferences": 1}
    assert deleted_before_replacement_registration == [False]
    assert set(assets) == {"asset-shared"}
    assert set(cache.items) == {"model-shared"}


def test_candidate_evaluation_collects_records_without_final_shard_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "final"
    args = builder.parse_args(
        ["--input_dir", str(tmp_path), "--output_dir", str(output_dir)]
    )
    tables = [
        {
            "source_table_id": table_id,
            "rows": [
                {
                    "row_id": 0,
                    "cells": [
                        {
                            "column_index": 0,
                            "column_name": "Entity",
                            "text": f"Entity {table_id}",
                            "wiki_title": f"Page {table_id}",
                        }
                    ],
                }
            ],
        }
        for table_id in ("recoverable", "failed")
    ]
    page_cache_path = tmp_path / "cache" / "wiki_pages.jsonl"
    image_cache_path = tmp_path / "cache" / "wiki_images.jsonl"
    wikipedia = SimpleNamespace(
        page_cache_path=page_cache_path,
        image_cache_path=image_cache_path,
        page_cache={
            f"Page {table_id}": {"wiki_title": f"Page {table_id}"}
            for table_id in ("recoverable", "failed")
        },
        image_cache={
            f"File:{table_id}.jpg": {"file_title": f"File:{table_id}.jpg"}
            for table_id in ("recoverable", "failed")
        },
    )
    cache = builder.ExtractionCache(tmp_path / "cache" / "model.jsonl")
    entity_records: dict[str, dict[str, object]] = {}
    wiki_to_entity_id: dict[str, str] = {}
    assets: dict[str, dict[str, object]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    registry = builder.CandidateMaterialRegistry(
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extraction_cache=cache,
    )
    context = builder.CandidateEvaluationContext(
        entity_records=entity_records,
        wiki_to_entity_id=wiki_to_entity_id,
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extractor=None,
        cache=cache,
        progress=None,
        concurrency_state=SimpleNamespace(),
        registry=registry,
    )

    def fake_batch_assets(**kwargs: object):
        entities = kwargs["entities"]
        writer = kwargs["asset_writer"]
        mapping: dict[str, list[str]] = {}
        for entity in entities:
            table_id = str(entity["wiki_title"]).removeprefix("Page ")
            asset_id = f"asset-{table_id}"
            writer.write_record(
                {
                    "asset_id": asset_id,
                    "entity_id": entity["entity_id"],
                    "entity_wiki_title": entity["wiki_title"],
                    "asset_type": "image",
                    "local_path": str(tmp_path / f"{table_id}.jpg"),
                    "image_url": f"https://images.example/{table_id}.jpg",
                    "metadata": {"file_title": f"File:{table_id}.jpg"},
                }
            )
            mapping[entity["entity_id"]] = [asset_id]
        return mapping, 0, 0, len(entities)

    def fake_build_table_join_records(**kwargs: object):
        source_table = kwargs["source_table"]
        assert isinstance(source_table, dict)
        table_id = str(source_table["source_table_id"])
        extraction_writer = kwargs["extraction_writer"]
        recovery_writer = kwargs["recovery_writer"]
        extraction_writer.write_record({"cache_key": f"model-{table_id}"})
        recovery_writer.write_record(
            {"evidence": {"extraction_cache_key": f"model-{table_id}"}}
        )
        query_tables = [{"table_id": f"query-{table_id}"}] if table_id == "recoverable" else []
        return query_tables, [], [], {"reason": "queryable" if query_tables else "failed"}

    monkeypatch.setattr(builder, "build_bridge_assets", fake_batch_assets)
    monkeypatch.setattr(builder, "build_table_join_records", fake_build_table_join_records)

    builder.prepare_candidate_batch(tables, context, args)
    evaluations = builder.evaluate_candidate_batch(tables, context, args)

    assert [evaluation.queryable for evaluation in evaluations] == [True, False]
    assert [evaluation.decision["reason"] for evaluation in evaluations] == [
        "queryable",
        "failed",
    ]
    for table_id in ("recoverable", "failed"):
        entity_id = wiki_to_entity_id[f"Page {table_id}"]
        assert registry.dependencies[table_id] == builder.CandidateDependencies(
            entities={entity_id},
            assets={f"asset-{table_id}"},
            paths={tmp_path / f"{table_id}.jpg"},
            urls={f"https://images.example/{table_id}.jpg"},
            page_keys={f"Page {table_id}"},
            imageinfo_keys={f"File:{table_id}.jpg"},
            model_keys={f"model-{table_id}"},
        )
    assert not (output_dir / "attribute_extractions").exists()
    assert not (output_dir / "evidence_recoveries").exists()


def _candidate_batch_task(
    *, cache_key: str, table_id: str, asset_type: str
) -> builder.ExtractionTask:
    return builder.ExtractionTask(
        order=0,
        cache_key=cache_key,
        source_table_id=table_id,
        source_row_id=0,
        entity_column_index=0,
        entity_column_name="Entity",
        entity={"entity_id": f"entity-{table_id}"},
        asset={"asset_id": f"asset-{cache_key}", "asset_type": asset_type},
        candidate_attribute_names=["State"],
    )


def _candidate_batch_context(
    tmp_path: Path, *, progress: object | None
) -> builder.CandidateEvaluationContext:
    cache = builder.ExtractionCache(tmp_path / "model.jsonl")
    assets: dict[str, dict[str, object]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    return builder.CandidateEvaluationContext(
        entity_records={},
        wiki_to_entity_id={},
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=None,
        extractor=None,
        cache=cache,
        progress=progress,  # type: ignore[arg-type]
        concurrency_state=SimpleNamespace(),
        registry=builder.CandidateMaterialRegistry(
            assets=assets,
            entity_to_assets=entity_to_assets,
            wikipedia_client=None,
            extraction_cache=cache,
        ),
    )


def test_candidate_evaluation_registers_and_precomputes_whole_batch_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--precompute_model_cache",
        ]
    )
    tables = [{"source_table_id": table_id, "rows": []} for table_id in ("a", "b")]
    events: list[str] = []
    registered: list[list[str]] = []

    class ProgressSpy:
        enabled = True

        def register(self, keys: object) -> None:
            registered.append(list(keys))  # type: ignore[arg-type]
            events.append("register")

    context = _candidate_batch_context(tmp_path, progress=ProgressSpy())
    task_map = {
        "a": [
            _candidate_batch_task(cache_key="shared-text", table_id="a", asset_type="text"),
            _candidate_batch_task(cache_key="image-a", table_id="a", asset_type="image"),
        ],
        "b": [
            _candidate_batch_task(cache_key="shared-text", table_id="b", asset_type="text"),
            _candidate_batch_task(cache_key="image-b", table_id="b", asset_type="image"),
        ],
    }

    monkeypatch.setattr(
        builder,
        "update_entities_from_table",
        lambda _records, _mapping, table: events.append(
            f"update:{table['source_table_id']}"
        ),
    )

    def collect_tasks(**kwargs: object) -> list[builder.ExtractionTask]:
        table = kwargs["source_table"]
        assert isinstance(table, dict)
        table_id = str(table["source_table_id"])
        events.append(f"collect:{table_id}")
        return task_map[table_id]

    calls: list[dict[str, list[builder.ExtractionTask]]] = []

    def precompute(**kwargs: object) -> dict[str, int]:
        tasks_by_kind = kwargs["tasks_by_kind"]
        assert isinstance(tasks_by_kind, dict)
        calls.append(tasks_by_kind)
        events.append("precompute")
        return {kind: len(tasks) for kind, tasks in tasks_by_kind.items()}

    def build_records(**kwargs: object):
        table = kwargs["source_table"]
        assert isinstance(table, dict)
        events.append(f"build:{table['source_table_id']}")
        return [], [], [], {"reason": "failed"}

    monkeypatch.setattr(builder, "collect_table_extraction_tasks", collect_tasks)
    monkeypatch.setattr(builder, "precompute_extraction_task_groups", precompute)
    monkeypatch.setattr(builder, "build_table_join_records", build_records)

    builder.evaluate_candidate_batch(tables, context, args)

    assert registered == [["shared-text", "image-a", "image-b"]]
    assert len(calls) == 1
    assert [task.cache_key for task in calls[0]["text"]] == ["shared-text"]
    assert [task.cache_key for task in calls[0]["image"]] == ["image-a", "image-b"]
    assert context.text_task_count == 1
    assert context.image_task_count == 2
    assert events == [
        "update:a",
        "update:b",
        "collect:a",
        "collect:b",
        "register",
        "precompute",
        "build:a",
        "build:b",
    ]


def test_candidate_evaluation_text_precompute_collects_only_batch_text_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--precompute_text_model_cache",
            "--no_model_progress",
        ]
    )
    tables = [{"source_table_id": table_id, "rows": []} for table_id in ("a", "b")]
    context = _candidate_batch_context(tmp_path, progress=None)
    asset_type_filters: list[set[str] | None] = []
    calls: list[dict[str, list[builder.ExtractionTask]]] = []
    monkeypatch.setattr(builder, "update_entities_from_table", lambda *_args: None)

    def collect_tasks(**kwargs: object) -> list[builder.ExtractionTask]:
        asset_types = kwargs["asset_types"]
        asset_type_filters.append(asset_types)  # type: ignore[arg-type]
        table = kwargs["source_table"]
        assert isinstance(table, dict)
        table_id = str(table["source_table_id"])
        return [
            _candidate_batch_task(
                cache_key="shared-text", table_id=table_id, asset_type="text"
            )
        ]

    def precompute(**kwargs: object) -> dict[str, int]:
        tasks_by_kind = kwargs["tasks_by_kind"]
        assert isinstance(tasks_by_kind, dict)
        calls.append(tasks_by_kind)
        return {kind: len(tasks) for kind, tasks in tasks_by_kind.items()}

    monkeypatch.setattr(builder, "collect_table_extraction_tasks", collect_tasks)
    monkeypatch.setattr(builder, "precompute_extraction_task_groups", precompute)
    monkeypatch.setattr(
        builder,
        "build_table_join_records",
        lambda **_kwargs: ([], [], [], {"reason": "failed"}),
    )

    builder.evaluate_candidate_batch(tables, context, args)

    assert asset_type_filters == [{"text"}, {"text"}]
    assert len(calls) == 1
    assert set(calls[0]) == {"text"}
    assert [task.cache_key for task in calls[0]["text"]] == ["shared-text"]
    assert context.text_task_count == 1
    assert context.image_task_count == 0


def test_candidate_text_precompute_marks_cache_hits_before_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--precompute_text_model_cache",
        ]
    )
    cached = _candidate_batch_task(
        cache_key="cached", table_id="table", asset_type="text"
    )
    pending = _candidate_batch_task(
        cache_key="pending", table_id="table", asset_type="text"
    )
    progress = builder.ModelAnalysisProgress(
        total=0, cached_keys=set(), enabled=False
    )
    progress.enabled = True
    context = _candidate_batch_context(tmp_path, progress=progress)
    context.cache.put(
        cached.cache_key,
        {
            "cache_key": cached.cache_key,
            "attributes": [{"name": "State", "value": "Alabama"}],
            "raw_response": '{"attributes":[]}',
            "error": "",
        },
    )
    monkeypatch.setattr(builder, "update_entities_from_table", lambda *_args: None)
    monkeypatch.setattr(
        builder,
        "collect_table_extraction_tasks",
        lambda **_kwargs: [cached, cached, pending],
    )

    def precompute(**kwargs: object) -> dict[str, int]:
        tasks_by_kind = kwargs["tasks_by_kind"]
        assert isinstance(tasks_by_kind, dict)
        assert progress.cached == 1
        assert progress.completed_keys == {cached.cache_key}
        assert [task.cache_key for task in tasks_by_kind["text"]] == [
            pending.cache_key
        ]
        return {"text": 1}

    monkeypatch.setattr(builder, "precompute_extraction_task_groups", precompute)
    monkeypatch.setattr(
        builder,
        "build_table_join_records",
        lambda **_kwargs: ([], [], [], {"reason": "failed"}),
    )

    builder.evaluate_candidate_batch(
        [{"source_table_id": "table", "rows": []}], context, args
    )

    assert progress.total == 2
    assert progress.cached == 1
    assert progress.planned_keys == {pending.cache_key}


def test_all_transient_candidate_batch_does_not_start_model_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_dir = tmp_path / "rounds"
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--precompute_model_cache",
            "--model_round_control_dir",
            str(control_dir),
        ]
    )
    text_task = _candidate_batch_task(
        cache_key="transient-text", table_id="table", asset_type="text"
    )
    image_task = _candidate_batch_task(
        cache_key="transient-image", table_id="table", asset_type="image"
    )
    progress = builder.ModelAnalysisProgress(
        total=0, cached_keys=set(), enabled=False
    )
    progress.enabled = True
    context = _candidate_batch_context(tmp_path, progress=progress)
    context.cache.put_transient(
        text_task.cache_key,
        {"cache_key": text_task.cache_key, "attributes": [], "error": "failed"},
    )
    context.cache.put_transient(
        image_task.cache_key,
        {
            "cache_key": image_task.cache_key,
            "attributes": [{"name": "State", "value": "Alabama"}],
            "error": "",
        },
    )
    monkeypatch.setattr(builder, "update_entities_from_table", lambda *_args: None)
    monkeypatch.setattr(
        builder,
        "collect_table_extraction_tasks",
        lambda **_kwargs: [text_task, image_task],
    )
    monkeypatch.setattr(
        builder,
        "build_table_join_records",
        lambda **_kwargs: ([], [], [], {"reason": "failed"}),
    )

    builder.evaluate_candidate_batch(
        [{"source_table_id": "table", "rows": []}], context, args
    )

    assert not control_dir.exists()
    assert context.text_task_count == 0
    assert context.image_task_count == 0
    assert progress.model == 2
    assert progress.errors == 1
    assert progress.completed_keys == {
        text_task.cache_key,
        image_task.cache_key,
    }


def test_candidate_evaluation_without_precompute_or_progress_skips_task_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--no_model_progress",
        ]
    )
    tables = [{"source_table_id": table_id, "rows": []} for table_id in ("a", "b")]
    context = _candidate_batch_context(tmp_path, progress=None)
    build_calls: list[str] = []
    monkeypatch.setattr(builder, "update_entities_from_table", lambda *_args: None)
    monkeypatch.setattr(
        builder,
        "collect_table_extraction_tasks",
        lambda **_kwargs: pytest.fail("unexpected extraction task scan"),
    )
    monkeypatch.setattr(
        builder,
        "precompute_extraction_task_groups",
        lambda **_kwargs: pytest.fail("unexpected model precompute"),
    )

    def build_records(**kwargs: object):
        table = kwargs["source_table"]
        assert isinstance(table, dict)
        build_calls.append(str(table["source_table_id"]))
        return [], [], [], {"reason": "failed"}

    monkeypatch.setattr(builder, "build_table_join_records", build_records)

    builder.evaluate_candidate_batch(tables, context, args)

    assert build_calls == ["a", "b"]


def test_candidate_batch_tasks_are_released_before_table_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--precompute_model_cache",
            "--no_model_progress",
        ]
    )
    context = _candidate_batch_context(tmp_path, progress=None)
    task_refs: list[weakref.ReferenceType[builder.ExtractionTask]] = []
    monkeypatch.setattr(builder, "update_entities_from_table", lambda *_args: None)

    def collect_tasks(**kwargs: object) -> list[builder.ExtractionTask]:
        table = kwargs["source_table"]
        assert isinstance(table, dict)
        task = _candidate_batch_task(
            cache_key=f"key-{table['source_table_id']}",
            table_id=str(table["source_table_id"]),
            asset_type="text",
        )
        task_refs.append(weakref.ref(task))
        return [task]

    def precompute(**kwargs: object) -> dict[str, int]:
        tasks_by_kind = kwargs["tasks_by_kind"]
        assert isinstance(tasks_by_kind, dict)
        return {kind: len(tasks) for kind, tasks in tasks_by_kind.items()}

    first_build = True

    def build_records(**_kwargs: object):
        nonlocal first_build
        if first_build:
            first_build = False
            gc.collect()
            assert task_refs
            assert all(task_ref() is None for task_ref in task_refs)
        return [], [], [], {"reason": "failed"}

    monkeypatch.setattr(builder, "collect_table_extraction_tasks", collect_tasks)
    monkeypatch.setattr(builder, "precompute_extraction_task_groups", precompute)
    monkeypatch.setattr(builder, "build_table_join_records", build_records)

    builder.evaluate_candidate_batch(
        [
            {"source_table_id": "a", "rows": []},
            {"source_table_id": "b", "rows": []},
        ],
        context,
        args,
    )


def test_batch_material_preparation_uses_batch_path_and_never_refetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        ["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "out")]
    )
    tables = [
        {
            "source_table_id": table_id,
            "rows": [
                {
                    "row_id": 0,
                    "cells": [
                        {
                            "column_index": index,
                            "column_name": "Entity",
                            "text": title,
                            "wiki_title": title,
                        }
                        for index, title in enumerate(titles)
                    ],
                }
            ],
        }
        for table_id, titles in (
            ("first", ("Page A", "Shared Page")),
            ("second", ("Shared Page", "Page B")),
            ("replacement", ("Shared Page", "Page Empty")),
        )
    ]
    wikipedia = SimpleNamespace(
        page_cache_path=tmp_path / "wiki_pages.jsonl",
        image_cache_path=tmp_path / "wiki_images.jsonl",
        page_cache={},
        image_cache={},
        api_failures=0,
    )
    cache = builder.ExtractionCache(tmp_path / "model.jsonl")
    assets: dict[str, dict[str, object]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    registry = builder.CandidateMaterialRegistry(
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extraction_cache=cache,
    )
    context = builder.CandidateEvaluationContext(
        entity_records={},
        wiki_to_entity_id={},
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extractor=None,
        cache=cache,
        progress=None,
        concurrency_state=SimpleNamespace(),
        registry=registry,
    )
    batches: list[list[str]] = []

    def fake_batch_assets(**kwargs: object):
        entities = kwargs["entities"]
        assert isinstance(entities, list)
        batches.append([str(entity["wiki_title"]) for entity in entities])
        writer = kwargs["asset_writer"]
        for entity in entities:
            title = str(entity["wiki_title"])
            wikipedia.page_cache[title] = {
                "wiki_title": title,
                "images": [{"title": f"File:{title}.jpg"}],
            }
            wikipedia.image_cache[f"File:{title}.jpg"] = {
                "file_title": f"File:{title}.jpg"
            }
            if title == "Page Empty":
                continue
            writer.write_record(
                {
                    "asset_id": f"asset-{title}",
                    "entity_id": entity["entity_id"],
                    "asset_type": "text",
                }
            )
        writer.flush()
        return (
            {
                entity["entity_id"]: []
                if entity["wiki_title"] == "Page Empty"
                else [f"asset-{entity['wiki_title']}"]
                for entity in entities
            },
            0,
            len(entities) - 1,
            0,
        )

    monkeypatch.setattr(builder, "build_bridge_assets", fake_batch_assets)
    monkeypatch.setattr(
        builder,
        "build_bridge_assets_for_entity",
        lambda **_kwargs: pytest.fail("per-entity material fallback was called"),
    )
    monkeypatch.setattr(
        builder,
        "build_table_join_records",
        lambda **_kwargs: ([], [], [], {"reason": "failed"}),
    )

    builder.prepare_candidate_batch(tables[:2], context, args)
    builder.prepare_candidate_batch([tables[2]], context, args)
    builder.prepare_candidate_batch([], context, args)
    builder.evaluate_candidate_batch(tables, context, args)

    assert batches == [["Page A", "Shared Page", "Page B"], ["Page Empty"]]
    empty_id = context.wiki_to_entity_id["Page Empty"]
    assert entity_to_assets[empty_id] == []
    assert registry.dependencies["replacement"].imageinfo_keys == {
        "File:Shared Page.jpg",
        "File:Page Empty.jpg",
    }


def test_candidate_evaluation_retains_rejected_imageinfo_after_active_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        ["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "final")]
    )
    tables = [
        {
            "source_table_id": table_id,
            "rows": [
                {
                    "row_id": 0,
                    "cells": [
                        {
                            "column_index": 0,
                            "column_name": "Entity",
                            "text": f"Entity {table_id}",
                            "wiki_title": page_title,
                        }
                    ],
                }
            ],
        }
        for table_id, page_title in (("discarded", "Page A"), ("retained", "Page B"))
    ]
    pages = {
        "Page A": {
            "wiki_title": "Page A",
            "images": [
                {"title": "File:Exclusive.jpg"},
                {"title": "File:Shared.jpg"},
            ],
        },
        "Page B": {
            "wiki_title": "Page B",
            "images": [{"title": "File:Shared.jpg"}],
        },
    }
    image_cache: dict[str, dict[str, object]] = {}

    def get_imageinfo(file_title: str) -> dict[str, object]:
        record = image_cache.setdefault(file_title, {"file_title": file_title})
        return record

    wikipedia = SimpleNamespace(
        page_cache_path=tmp_path / "cache" / "wiki_pages.jsonl",
        image_cache_path=tmp_path / "cache" / "wiki_images.jsonl",
        page_cache=dict(pages),
        image_cache=image_cache,
        api_failures=0,
        get_page=pages.get,
        get_imageinfo=get_imageinfo,
        download_image=lambda _imageinfo, _asset_id: None,
    )
    cache = builder.ExtractionCache(tmp_path / "cache" / "model.jsonl")
    assets: dict[str, dict[str, object]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    registry = builder.CandidateMaterialRegistry(
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extraction_cache=cache,
    )
    context = builder.CandidateEvaluationContext(
        entity_records={},
        wiki_to_entity_id={},
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extractor=None,
        cache=cache,
        progress=None,
        concurrency_state=SimpleNamespace(),
        registry=registry,
    )
    monkeypatch.setattr(
        builder,
        "build_table_join_records",
        lambda **_kwargs: ([], [], [], {"reason": "failed"}),
    )

    builder.prepare_candidate_batch(tables, context, args)
    builder.evaluate_candidate_batch(tables, context, args)

    assert assets == {}
    assert registry.dependencies["discarded"].imageinfo_keys == {
        "File:Exclusive.jpg",
        "File:Shared.jpg",
    }
    assert registry.dependencies["retained"].imageinfo_keys == {"File:Shared.jpg"}

    stats = registry.sweep({"retained"})

    assert stats.imageinfo_records_removed == 0
    assert stats.shared_dependencies_protected >= 1
    assert image_cache.keys() == {"File:Exclusive.jpg", "File:Shared.jpg"}


def test_candidate_evaluation_reuses_rejected_imageinfo_for_shared_entity_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        ["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "final")]
    )
    tables = [
        {
            "source_table_id": table_id,
            "rows": [
                {
                    "row_id": 0,
                    "cells": [
                        {
                            "column_index": 0,
                            "column_name": "Entity",
                            "text": "Shared entity",
                            "wiki_title": "Shared Page",
                        }
                    ],
                }
            ],
        }
        for table_id in ("discarded", "retained")
    ]
    pages = {
        "Shared Page": {
            "wiki_title": "Shared Page",
            "images": [{"title": "File:Shared.jpg"}],
        }
    }
    image_cache: dict[str, dict[str, object]] = {}

    def get_imageinfo(file_title: str) -> dict[str, object]:
        record = image_cache.setdefault(file_title, {"file_title": file_title})
        return record

    wikipedia = SimpleNamespace(
        page_cache_path=tmp_path / "cache" / "wiki_pages.jsonl",
        image_cache_path=tmp_path / "cache" / "wiki_images.jsonl",
        page_cache=dict(pages),
        image_cache=image_cache,
        api_failures=0,
        get_page=pages.get,
        get_imageinfo=get_imageinfo,
        download_image=lambda _imageinfo, _asset_id: None,
    )
    cache = builder.ExtractionCache(tmp_path / "cache" / "model.jsonl")
    assets: dict[str, dict[str, object]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    registry = builder.CandidateMaterialRegistry(
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extraction_cache=cache,
    )
    context = builder.CandidateEvaluationContext(
        entity_records={},
        wiki_to_entity_id={},
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extractor=None,
        cache=cache,
        progress=None,
        concurrency_state=SimpleNamespace(),
        registry=registry,
    )
    monkeypatch.setattr(
        builder,
        "build_table_join_records",
        lambda **_kwargs: ([], [], [], {"reason": "failed"}),
    )

    builder.prepare_candidate_batch(tables, context, args)
    builder.evaluate_candidate_batch(tables, context, args)

    assert assets == {}
    assert registry.dependencies["retained"].imageinfo_keys == {"File:Shared.jpg"}

    stats = registry.sweep({"retained"})

    assert stats.imageinfo_records_removed == 0
    assert stats.shared_dependencies_protected >= 1
    assert image_cache.keys() == {"File:Shared.jpg"}


def write_keyed_cache(path: Path, key_name: str, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({key_name: key, "value": key}) + "\n" for key in keys),
        encoding="utf-8",
    )


def cleanup_registry(tmp_path: Path) -> tuple[
    builder.CandidateMaterialRegistry,
    dict[str, dict[str, str]],
    dict[str, list[str]],
    SimpleNamespace,
    builder.ExtractionCache,
    Path,
    Path,
]:
    exclusive_image = tmp_path / "exclusive.jpg"
    shared_image = tmp_path / "shared.jpg"
    exclusive_image.write_bytes(b"exclusive-image")
    shared_image.write_bytes(b"shared-image")

    page_path = tmp_path / "wikipedia" / "wiki_pages.jsonl"
    image_path = tmp_path / "wikipedia" / "wiki_images.jsonl"
    model_path = tmp_path / "model_attribute_extractions.jsonl"
    write_keyed_cache(page_path, "wiki_title", ["Page A", "Page shared"])
    write_keyed_cache(image_path, "file_title", ["File:A.jpg", "File:Shared.jpg"])
    write_keyed_cache(model_path, "cache_key", ["model-a", "model-shared"])

    wikipedia = SimpleNamespace(
        page_cache_path=page_path,
        image_cache_path=image_path,
        page_cache={
            "Page A": {"wiki_title": "Page A", "value": "Page A"},
            "Page shared": {"wiki_title": "Page shared", "value": "Page shared"},
        },
        image_cache={
            "File:A.jpg": {"file_title": "File:A.jpg", "value": "File:A.jpg"},
            "File:Shared.jpg": {
                "file_title": "File:Shared.jpg",
                "value": "File:Shared.jpg",
            },
        },
    )
    model_cache = builder.ExtractionCache(model_path)
    assets = {
        "asset-a": {
            "asset_id": "asset-a",
            "local_path": str(exclusive_image),
            "image_url": "https://images.example/a.jpg",
        },
        "asset-shared": {
            "asset_id": "asset-shared",
            "local_path": str(shared_image),
            "image_url": "https://images.example/shared.jpg",
        },
    }
    entity_to_assets = {
        "entity-a": ["asset-a", "asset-shared"],
        "entity-shared": ["asset-a", "asset-shared"],
    }
    registry = builder.CandidateMaterialRegistry(
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia,
        extraction_cache=model_cache,
    )
    registry.register(
        "A",
        builder.CandidateDependencies(
            entities=frozenset({"entity-a", "entity-shared"}),
            assets=frozenset({"asset-a", "asset-shared"}),
            paths=frozenset({exclusive_image, shared_image}),
            urls=frozenset(
                {"https://images.example/a.jpg", "https://images.example/shared.jpg"}
            ),
            page_keys=frozenset({"Page A", "Page shared"}),
            imageinfo_keys=frozenset({"File:A.jpg", "File:Shared.jpg"}),
            model_keys=frozenset({"model-a", "model-shared"}),
        ),
    )
    registry.register(
        "B",
        builder.CandidateDependencies(
            entities=frozenset({"entity-shared"}),
            assets=frozenset({"asset-shared"}),
            paths=frozenset({shared_image}),
            urls=frozenset({"https://images.example/shared.jpg"}),
            page_keys=frozenset({"Page shared"}),
            imageinfo_keys=frozenset({"File:Shared.jpg"}),
            model_keys=frozenset({"model-shared"}),
        ),
    )
    return (
        registry,
        assets,
        entity_to_assets,
        wikipedia,
        model_cache,
        exclusive_image,
        shared_image,
    )


def test_discard_prunes_active_material_but_retains_persistent_caches(
    tmp_path: Path,
) -> None:
    (
        registry,
        assets,
        entity_to_assets,
        wikipedia,
        model_cache,
        exclusive_image,
        shared_image,
    ) = cleanup_registry(tmp_path)
    stats = registry.discard("A")

    assert assets == {"asset-shared": assets["asset-shared"]}
    assert entity_to_assets == {"entity-shared": ["asset-shared"]}
    assert wikipedia.page_cache.keys() == {"Page A", "Page shared"}
    assert wikipedia.image_cache.keys() == {"File:A.jpg", "File:Shared.jpg"}
    assert model_cache.items.keys() == {"model-a", "model-shared"}
    assert exclusive_image.exists()
    assert shared_image.exists()
    assert stats == builder.CacheCleanupStats(
        entities_removed=1,
        assets_removed=1,
        shared_dependencies_protected=7,
    )
    assert [record["wiki_title"] for record in builder.iter_jsonl_records([wikipedia.page_cache_path])] == [
        "Page A",
        "Page shared",
    ]
    assert [record["file_title"] for record in builder.iter_jsonl_records([wikipedia.image_cache_path])] == [
        "File:A.jpg",
        "File:Shared.jpg",
    ]
    assert [record["cache_key"] for record in builder.iter_jsonl_records([model_cache.path])] == [
        "model-a",
        "model-shared",
    ]
    restarted_cache = builder.ExtractionCache(model_cache.path)
    assert restarted_cache.items.keys() == {"model-a", "model-shared"}


def test_sweep_prunes_orphans_without_compacting_persistent_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        registry,
        assets,
        entity_to_assets,
        wikipedia,
        model_cache,
        exclusive_image,
        shared_image,
    ) = cleanup_registry(tmp_path)
    second_image = tmp_path / "second.jpg"
    second_image.write_bytes(b"second-image")
    assets["asset-c"] = {
        "asset_id": "asset-c",
        "local_path": str(second_image),
        "image_url": "https://images.example/c.jpg",
    }
    entity_to_assets["entity-c"] = ["asset-c", "asset-shared"]
    wikipedia.page_cache["Page C"] = {"wiki_title": "Page C"}
    wikipedia.image_cache["File:C.jpg"] = {"file_title": "File:C.jpg"}
    model_cache.items["model-c"] = {"cache_key": "model-c"}
    registry.register(
        "C",
        builder.CandidateDependencies(
            entities={"entity-c", "entity-shared"},
            assets={"asset-c", "asset-shared"},
            paths={second_image, shared_image},
            urls={
                "https://images.example/c.jpg",
                "https://images.example/shared.jpg",
            },
            page_keys={"Page C", "Page shared"},
            imageinfo_keys={"File:C.jpg", "File:Shared.jpg"},
            model_keys={"model-c", "model-shared"},
        ),
    )
    cache_contents = {
        wikipedia.page_cache_path: wikipedia.page_cache_path.read_bytes(),
        wikipedia.image_cache_path: wikipedia.image_cache_path.read_bytes(),
        model_cache.path: model_cache.path.read_bytes(),
    }

    def fail_compaction(_path: Path, _records: object) -> None:
        raise AssertionError("persistent caches must not be compacted during selection")

    monkeypatch.setattr(builder, "compact_keyed_jsonl", fail_compaction)

    stats = registry.sweep({"B"})

    assert stats == builder.CacheCleanupStats(
        entities_removed=2,
        assets_removed=2,
        shared_dependencies_protected=7,
    )
    assert set(registry.dependencies) == {"B"}
    assert set(assets) == {"asset-shared"}
    assert entity_to_assets == {"entity-shared": ["asset-shared"]}
    assert wikipedia.page_cache.keys() == {"Page A", "Page C", "Page shared"}
    assert wikipedia.image_cache.keys() == {
        "File:A.jpg",
        "File:C.jpg",
        "File:Shared.jpg",
    }
    assert model_cache.items.keys() == {"model-a", "model-c", "model-shared"}
    assert exclusive_image.exists()
    assert second_image.exists()
    assert shared_image.exists()
    for path, original_contents in cache_contents.items():
        assert path.read_bytes() == original_contents


def test_build_dataset_materializes_only_settled_replacement_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    write_entitables_file(input_dir / "tables.json", ["t0", "t1", "t2"])
    candidate_tables = [
        {
            "source_table_id": table_id,
            "page_title": table_id,
            "rows": [
                {
                    "row_id": 0,
                    "cells": [
                        {
                            "column_index": 0,
                            "column_name": "Entity",
                            "text": table_id,
                            "wiki_title": f"Page {table_id}",
                        }
                    ],
                }
            ],
        }
        for table_id in ("t0", "t1", "t2")
    ]

    candidate_iterator_closed: list[bool] = []
    candidate_progresses: list[builder.ModelAnalysisProgress | None] = []
    materialization_progresses: list[builder.ModelAnalysisProgress | None] = []

    def fake_candidates(
        _input_dir: Path,
        _args: object,
        counters: builder.SourceCandidateCounters,
    ):
        counters.processed_tables = 3
        try:
            yield from candidate_tables
        finally:
            candidate_iterator_closed.append(True)

    def fake_evaluate(
        tables: list[dict[str, object]],
        context: builder.CandidateEvaluationContext,
        _args: object,
    ) -> list[builder.CandidateEvaluation]:
        candidate_progresses.append(context.progress)
        for table in tables:
            builder.update_entities_from_table(
                context.entity_records, context.wiki_to_entity_id, table
            )
            table_id = str(table["source_table_id"])
            entity_id = context.wiki_to_entity_id[f"Page {table_id}"]
            asset_id = f"asset-{table_id}"
            context.assets[asset_id] = {
                "asset_id": asset_id,
                "entity_id": entity_id,
                "asset_type": "text",
            }
            context.entity_to_assets[entity_id] = [asset_id]
            context.registry.register(
                table_id,
                builder.CandidateDependencies(
                    entities={entity_id}, assets={asset_id}
                ),
            )
        return [
            builder.CandidateEvaluation(
                source_table=table,
                queryable=table["source_table_id"] != "t0",
                decision={"reason": "candidate-only"},
            )
            for table in tables
        ]

    def fake_final_records(**kwargs: object):
        materialization_progresses.append(kwargs["progress"])
        source_table = kwargs["source_table"]
        assert isinstance(source_table, dict)
        table_id = str(source_table["source_table_id"])
        queryable = table_id != "t0"
        query_id = f"query-{table_id}"
        target_id = f"target-{table_id}"
        return (
            [{"table_id": query_id, "source_table_id": table_id}] if queryable else [],
            [{"table_id": target_id, "source_table_id": table_id}],
            [
                {
                    "query_table_id": query_id,
                    "candidate_table_id": target_id,
                    "relevance": 1,
                    "source_table_id": table_id,
                }
            ]
            if queryable
            else [],
            {"reason": "queryable" if queryable else "failed"},
        )

    monkeypatch.setattr(builder, "iter_random_source_tables", fake_candidates)
    monkeypatch.setattr(builder, "evaluate_candidate_batch", fake_evaluate)
    monkeypatch.setattr(builder, "build_table_join_records", fake_final_records)
    monkeypatch.setattr(builder, "LocalAttributeExtractor", lambda _args: None)
    real_sweep = builder.CandidateMaterialRegistry.sweep

    def assert_closed_before_sweep(
        registry: builder.CandidateMaterialRegistry,
        final_table_ids: object,
    ) -> builder.CacheCleanupStats:
        assert candidate_iterator_closed == [True]
        return real_sweep(registry, final_table_ids)

    monkeypatch.setattr(
        builder.CandidateMaterialRegistry,
        "sweep",
        assert_closed_before_sweep,
    )

    args = builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--max_source_tables",
            "2",
            "--unrecoverable_replacement_rounds",
            "1",
            "--unrecoverable_drop_probability",
            "1",
            "--no_wikipedia",
            "--no_model_progress",
            "--records_per_shard",
            "1",
        ]
    )

    stats = builder.build_dataset(args)

    assert candidate_iterator_closed == [True]
    assert candidate_progresses
    assert all(progress is None for progress in candidate_progresses)
    assert materialization_progresses
    assert all(progress is None for progress in materialization_progresses)

    source_rows = list(
        builder.iter_jsonl_records(sorted((output_dir / "source_tables").glob("*.jsonl")))
    )
    decisions = list(
        builder.iter_jsonl_records([output_dir / "table_queryability_decisions.jsonl"])
    )
    bridge_assets = list(
        builder.iter_jsonl_records(sorted((output_dir / "bridge_assets").glob("*.jsonl")))
    )
    qrels = list(builder.iter_jsonl_records([output_dir / "qrels.jsonl"]))
    splits = json.loads((output_dir / "splits.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )

    assert [row["source_table_id"] for row in source_rows] == ["t2", "t1"]
    assert {row["source_table_id"] for row in decisions} == {"t1", "t2"}
    assert {row["asset_id"] for row in bridge_assets} == {"asset-t1", "asset-t2"}
    assert all(row.get("source_table_id") != "t0" for row in qrels)
    assert all(
        "t0" not in payload.get("source_table_ids", [])
        for payload in splits.values()
        if isinstance(payload, dict)
    )
    assert manifest["artifacts"]["source_tables"]["total_records"] == len(source_rows)
    assert sum(
        shard["records"]
        for shard in manifest["artifacts"]["source_tables"]["shards"]
    ) == len(source_rows)
    assert stats["sampling_seed"] == args.seed
    assert stats["unrecoverable_replacement_rounds"] == 1
    assert stats["unrecoverable_drop_probability"] == 1.0
    assert stats["unrecoverable_replacement_scope"] == "all_current_failed_slots"
    assert stats["discarded_candidate_cache_policy"] == "retain_persistent_cache"
    assert stats["replacement_selection"] == {
        "rounds": [
            {
                "round_index": 0,
                "evaluated": 2,
                "unrecoverable": 1,
                "discarded": 1,
                "retained_failed": 0,
                "replacements": 1,
            },
            {
                "round_index": 1,
                "evaluated": 1,
                "unrecoverable": 0,
                "discarded": 0,
                "retained_failed": 0,
                "replacements": 0,
            },
        ],
        "candidates_consumed": 3,
        "candidate_exhausted": False,
        "unfilled_slots": 0,
    }
    assert set(stats["cleanup"]) == set(builder.CacheCleanupStats.__dataclass_fields__)
    assert manifest["source_sampling"] == {
        "mode": "seeded_random_file_and_table_order",
        "seed": args.seed,
        "entity_column_policy": "require_candidate_before_global_sampling",
        "unrecoverable_replacement_rounds": 1,
        "unrecoverable_drop_probability": 1.0,
        "replacement_scope": "all_current_failed_slots",
        "discarded_candidate_cache_policy": "retain_persistent_cache",
    }


def test_build_dataset_closes_model_progress_when_intermediate_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    progress_instances: list[object] = []

    class ProgressSpy:
        enabled = False

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            progress_instances.append(self)

        def close(self) -> None:
            self.close_calls += 1

    def fake_selection(**kwargs: object) -> builder.ReplacementSelection:
        callback = kwargs["on_initial_batch_prepared"]
        assert callable(callback)
        callback([])
        return builder.ReplacementSelection([], [], 0, True, 0)

    monkeypatch.setattr(builder, "ModelAnalysisProgress", ProgressSpy)
    monkeypatch.setattr(builder, "run_replacement_rounds", fake_selection)
    monkeypatch.setattr(builder, "LocalAttributeExtractor", lambda _args: None)
    monkeypatch.setattr(
        builder,
        "write_sharded_jsonl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    args = builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(tmp_path / "output"),
            "--max_source_tables",
            "0",
            "--no_wikipedia",
        ]
    )

    with pytest.raises(RuntimeError, match="write failed"):
        builder.build_dataset(args)

    assert len(progress_instances) == 1
    assert progress_instances[0].close_calls == 1


def test_candidate_evaluation_skips_extra_task_scan_when_progress_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "output"),
            "--no_model_progress",
        ]
    )
    cache = builder.ExtractionCache(tmp_path / "model.jsonl")
    assets: dict[str, dict[str, object]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    context = builder.CandidateEvaluationContext(
        entity_records={},
        wiki_to_entity_id={},
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=None,
        extractor=None,
        cache=cache,
        progress=None,
        concurrency_state=SimpleNamespace(),
        registry=builder.CandidateMaterialRegistry(
            assets=assets,
            entity_to_assets=entity_to_assets,
            wikipedia_client=None,
            extraction_cache=cache,
        ),
    )
    monkeypatch.setattr(
        builder,
        "collect_table_extraction_tasks",
        lambda **_kwargs: pytest.fail("unexpected duplicate extraction task scan"),
    )
    monkeypatch.setattr(
        builder,
        "build_table_join_records",
        lambda **_kwargs: ([], [], [], {"reason": "failed"}),
    )

    evaluations = builder.evaluate_candidate_batch(
        [{"source_table_id": "table", "rows": []}], context, args
    )

    assert len(evaluations) == 1


def test_hidden_text_precompute_collects_only_text_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "output"),
            "--no_model_progress",
            "--precompute_text_model_cache",
        ]
    )
    cache = builder.ExtractionCache(tmp_path / "model.jsonl")
    assets: dict[str, dict[str, object]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    context = builder.CandidateEvaluationContext(
        entity_records={},
        wiki_to_entity_id={},
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=None,
        extractor=None,
        cache=cache,
        progress=None,
        concurrency_state=builder.ModelConcurrencyState(1, 1),
        registry=builder.CandidateMaterialRegistry(
            assets=assets,
            entity_to_assets=entity_to_assets,
            wikipedia_client=None,
            extraction_cache=cache,
        ),
    )
    collected_asset_types: list[set[str] | None] = []

    def fake_collect(**kwargs: object) -> list[builder.ExtractionTask]:
        collected_asset_types.append(kwargs.get("asset_types"))
        return []

    monkeypatch.setattr(builder, "collect_table_extraction_tasks", fake_collect)
    monkeypatch.setattr(
        builder,
        "build_table_join_records",
        lambda **_kwargs: ([], [], [], {"reason": "failed"}),
    )

    builder.evaluate_candidate_batch(
        [{"source_table_id": "table", "rows": []}], context, args
    )

    assert collected_asset_types == [{"text"}]


def test_candidate_evaluation_registers_real_extraction_tasks_for_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = builder.parse_args(
        ["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "output")]
    )
    source_table = {
        "source_table_id": "table",
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "State"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "Entity",
                        "text": "Alpha",
                        "wiki_title": "Alpha",
                    },
                    {"column_index": 1, "column_name": "State", "text": "Alabama"},
                ],
            }
        ],
        "metadata": {"candidate_entity_columns": [0]},
    }
    entity_records: dict[str, dict[str, object]] = {}
    wiki_to_entity_id: dict[str, str] = {}
    builder.update_entities_from_table(entity_records, wiki_to_entity_id, source_table)
    entity_id = wiki_to_entity_id["Alpha"]
    assets = {
        "asset": {
            "asset_id": "asset",
            "asset_type": "text",
            "entity_id": entity_id,
            "content": "Alpha is in Alabama.",
        }
    }
    entity_to_assets = {entity_id: ["asset"]}
    cache = builder.ExtractionCache(tmp_path / "model.jsonl")
    progress = builder.ModelAnalysisProgress(total=0, cached_keys=set(), enabled=False)
    progress.enabled = True
    context = builder.CandidateEvaluationContext(
        entity_records=entity_records,
        wiki_to_entity_id=wiki_to_entity_id,
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=None,
        extractor=None,
        cache=cache,
        progress=progress,
        concurrency_state=SimpleNamespace(),
        registry=builder.CandidateMaterialRegistry(
            assets=assets,
            entity_to_assets=entity_to_assets,
            wikipedia_client=None,
            extraction_cache=cache,
        ),
    )
    monkeypatch.setattr(
        builder,
        "build_table_join_records",
        lambda **_kwargs: ([], [], [], {"reason": "failed"}),
    )

    builder.evaluate_candidate_batch([source_table], context, args)

    assert progress.total == 1
    assert len(progress.planned_keys) == 1


@pytest.mark.parametrize("callable_close", [True, False])
def test_build_dataset_closes_candidates_when_selection_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    callable_close: bool,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    class CandidateIterator:
        def __init__(self) -> None:
            self.closed = False
            if not callable_close:
                self.close = "not-callable"

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self) -> None:
            self.closed = True

    candidates = CandidateIterator()
    monkeypatch.setattr(
        builder,
        "iter_random_source_tables",
        lambda _input_dir, _args, _counters: candidates,
    )

    def fail_selection(**_kwargs: object):
        raise RuntimeError("selection failed")

    monkeypatch.setattr(builder, "run_replacement_rounds", fail_selection)
    args = builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--max_source_tables",
            "1",
            "--no_wikipedia",
            "--no_model_progress",
        ]
    )

    with pytest.raises(RuntimeError, match="selection failed"):
        builder.build_dataset(args)

    assert candidates.closed is callable_close


def test_candidate_selection_and_materialization_share_first_seen_entity_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    candidate_tables = [
        {
            "source_table_id": table_id,
            "page_title": table_id,
            "rows": [
                {
                    "row_id": 0,
                    "cells": [
                        {
                            "column_index": 0,
                            "column_name": "Entity",
                            "text": wiki_title,
                            "wiki_title": wiki_title,
                        }
                    ],
                }
            ],
        }
        for table_id, wiki_title in (("first", "Page Z"), ("second", "Page A"))
    ]

    monkeypatch.setattr(
        builder,
        "iter_random_source_tables",
        lambda _input_dir, _args, _counters: iter(candidate_tables),
    )

    def fake_batch_assets(**kwargs: object):
        entities = kwargs["entities"]
        writer = kwargs["asset_writer"]
        mapping: dict[str, list[str]] = {}
        for entity in entities:
            entity_id = entity["entity_id"]
            asset_id = f"asset-{entity_id}"
            writer.write_record(
                {
                    "asset_id": asset_id,
                    "entity_id": entity_id,
                    "entity_wiki_title": entity["wiki_title"],
                    "asset_type": "text",
                }
            )
            mapping[entity_id] = [asset_id]
        return mapping, 0, len(entities), 0

    def fake_join_records(**kwargs: object):
        source_table = kwargs["source_table"]
        wiki_to_entity_id = kwargs["wiki_to_entity_id"]
        entity_to_assets = kwargs["entity_to_assets"]
        assert isinstance(source_table, dict)
        wiki_title = source_table["rows"][0]["cells"][0]["wiki_title"]
        entity_id = wiki_to_entity_id[wiki_title]
        queryable = bool(entity_to_assets.get(entity_id))
        table_id = str(source_table["source_table_id"])
        return (
            [{"table_id": f"query-{table_id}", "source_table_id": table_id}]
            if queryable
            else [],
            [],
            [],
            {"source_table_id": table_id, "reason": "queryable" if queryable else "failed"},
        )

    monkeypatch.setattr(builder, "build_bridge_assets", fake_batch_assets)
    monkeypatch.setattr(
        builder,
        "WikipediaClient",
        lambda **_kwargs: SimpleNamespace(
            page_cache_path=tmp_path / "pages.jsonl",
            image_cache_path=tmp_path / "images.jsonl",
            page_cache={},
            image_cache={},
            api_failures=0,
        ),
    )
    monkeypatch.setattr(builder, "build_table_join_records", fake_join_records)
    monkeypatch.setattr(builder, "LocalAttributeExtractor", lambda _args: None)

    args = builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--max_source_tables",
            "2",
            "--max_entities",
            "1",
            "--no_model_progress",
        ]
    )

    stats = builder.build_dataset(args)

    bridge_assets = list(
        builder.iter_jsonl_records(sorted((output_dir / "bridge_assets").glob("*.jsonl")))
    )
    decisions = list(
        builder.iter_jsonl_records([output_dir / "table_queryability_decisions.jsonl"])
    )
    assert [asset["entity_wiki_title"] for asset in bridge_assets] == ["Page Z"]
    assert {
        decision["source_table_id"]: decision["reason"] for decision in decisions
    } == {"first": "queryable", "second": "failed"}
    assert stats["queryable_source_tables"] == 1

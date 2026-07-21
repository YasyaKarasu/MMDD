import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_failed_table_is_retained_when_draw_equals_probability() -> None:
    tables = replacement_tables()
    discarded: list[str] = []

    selection = builder.run_replacement_rounds(
        candidate_tables=iter(tables),
        target_count=1,
        policy=builder.ReplacementPolicy(rounds=2, drop_probability=0.5),
        rng=StubRandom([0.5]),
        evaluate_batch=evaluator_for({"t0": False}),
        discard_tables=lambda table_ids: discarded.extend(table_ids),
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

    def fake_assets_for_entity(**kwargs: object) -> list[dict[str, object]]:
        entity = kwargs["entity"]
        assert isinstance(entity, dict)
        table_id = str(entity["wiki_title"]).removeprefix("Page ")
        return [
            {
                "asset_id": f"asset-{table_id}",
                "entity_id": entity["entity_id"],
                "entity_wiki_title": entity["wiki_title"],
                "asset_type": "image",
                "local_path": str(tmp_path / f"{table_id}.jpg"),
                "image_url": f"https://images.example/{table_id}.jpg",
                "metadata": {"file_title": f"File:{table_id}.jpg"},
            }
        ]

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

    monkeypatch.setattr(builder, "build_bridge_assets_for_entity", fake_assets_for_entity)
    monkeypatch.setattr(builder, "build_table_join_records", fake_build_table_join_records)

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


def test_candidate_evaluation_tracks_rejected_imageinfo_for_shared_cleanup(
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

    builder.evaluate_candidate_batch(tables, context, args)

    assert assets == {}
    assert registry.dependencies["discarded"].imageinfo_keys == {
        "File:Exclusive.jpg",
        "File:Shared.jpg",
    }
    assert registry.dependencies["retained"].imageinfo_keys == {"File:Shared.jpg"}

    stats = registry.sweep({"retained"})

    assert stats.imageinfo_records_removed == 1
    assert stats.shared_dependencies_protected >= 1
    assert image_cache.keys() == {"File:Shared.jpg"}
    assert [
        record["file_title"]
        for record in builder.iter_jsonl_records([wikipedia.image_cache_path])
    ] == ["File:Shared.jpg"]


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


def test_discard_removes_exclusive_material_and_preserves_shared_dependencies(
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
    exclusive_bytes = exclusive_image.stat().st_size

    stats = registry.discard("A")

    assert assets == {"asset-shared": assets["asset-shared"]}
    assert entity_to_assets == {"entity-shared": ["asset-shared"]}
    assert wikipedia.page_cache.keys() == {"Page shared"}
    assert wikipedia.image_cache.keys() == {"File:Shared.jpg"}
    assert model_cache.items.keys() == {"model-shared"}
    assert not exclusive_image.exists()
    assert shared_image.exists()
    assert stats == builder.CacheCleanupStats(
        entities_removed=1,
        assets_removed=1,
        page_records_removed=1,
        imageinfo_records_removed=1,
        model_records_removed=1,
        image_files_removed=1,
        image_bytes_removed=exclusive_bytes,
        shared_dependencies_protected=7,
    )
    assert [record["wiki_title"] for record in builder.iter_jsonl_records([wikipedia.page_cache_path])] == [
        "Page shared"
    ]
    assert [record["file_title"] for record in builder.iter_jsonl_records([wikipedia.image_cache_path])] == [
        "File:Shared.jpg"
    ]
    assert [record["cache_key"] for record in builder.iter_jsonl_records([model_cache.path])] == [
        "model-shared"
    ]


def test_sweep_batches_orphans_compacts_once_and_aggregates_cleanup(
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
    calls: list[Path] = []
    original_compact = builder.compact_keyed_jsonl

    def count_compaction(path: Path, records: object) -> None:
        calls.append(path)
        original_compact(path, records)

    monkeypatch.setattr(builder, "compact_keyed_jsonl", count_compaction)
    removed_bytes = exclusive_image.stat().st_size + second_image.stat().st_size

    stats = registry.sweep({"B"})

    assert calls == [
        wikipedia.page_cache_path,
        wikipedia.image_cache_path,
        model_cache.path,
    ]
    assert stats == builder.CacheCleanupStats(
        entities_removed=2,
        assets_removed=2,
        page_records_removed=2,
        imageinfo_records_removed=2,
        model_records_removed=2,
        image_files_removed=2,
        image_bytes_removed=removed_bytes,
        shared_dependencies_protected=7,
    )
    assert set(registry.dependencies) == {"B"}
    assert set(assets) == {"asset-shared"}
    assert entity_to_assets == {"entity-shared": ["asset-shared"]}
    assert wikipedia.page_cache.keys() == {"Page shared"}
    assert wikipedia.image_cache.keys() == {"File:Shared.jpg"}
    assert model_cache.items.keys() == {"model-shared"}
    assert not exclusive_image.exists()
    assert not second_image.exists()
    assert shared_image.exists()


def test_discard_many_continues_after_one_cache_compaction_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _, _, wikipedia, model_cache, _, _ = cleanup_registry(tmp_path)
    calls: list[Path] = []
    original_compact = builder.compact_keyed_jsonl

    def fail_page_compaction(path: Path, records: object) -> None:
        calls.append(path)
        if path == wikipedia.page_cache_path:
            raise OSError("injected page compaction failure")
        original_compact(path, records)

    monkeypatch.setattr(builder, "compact_keyed_jsonl", fail_page_compaction)

    stats = registry.discard_many(["A"])

    assert calls == [
        wikipedia.page_cache_path,
        wikipedia.image_cache_path,
        model_cache.path,
    ]
    assert stats.errors == 1
    assert [
        record["file_title"]
        for record in builder.iter_jsonl_records([wikipedia.image_cache_path])
    ] == ["File:Shared.jpg"]
    assert [
        record["cache_key"]
        for record in builder.iter_jsonl_records([model_cache.path])
    ] == ["model-shared"]


def test_sweep_counts_unlink_error_and_continues_cache_compactions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        registry,
        _assets,
        _entity_to_assets,
        wikipedia,
        model_cache,
        exclusive_image,
        shared_image,
    ) = cleanup_registry(tmp_path)
    original_unlink = Path.unlink

    def fail_exclusive_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == exclusive_image:
            raise OSError("injected unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_exclusive_unlink)

    stats = registry.sweep({"B"})

    assert stats.page_records_removed == 1
    assert stats.imageinfo_records_removed == 1
    assert stats.model_records_removed == 1
    assert stats.image_files_removed == 0
    assert stats.image_bytes_removed == 0
    assert stats.shared_dependencies_protected == 7
    assert stats.errors == 1
    assert exclusive_image.exists()
    assert shared_image.exists()
    assert wikipedia.page_cache.keys() == {"Page shared"}
    assert wikipedia.image_cache.keys() == {"File:Shared.jpg"}
    assert model_cache.items.keys() == {"model-shared"}
    assert [record["wiki_title"] for record in builder.iter_jsonl_records([wikipedia.page_cache_path])] == [
        "Page shared"
    ]
    assert [record["file_title"] for record in builder.iter_jsonl_records([wikipedia.image_cache_path])] == [
        "File:Shared.jpg"
    ]
    assert [record["cache_key"] for record in builder.iter_jsonl_records([model_cache.path])] == [
        "model-shared"
    ]


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

    def fake_candidates(
        _input_dir: Path,
        _args: object,
        counters: builder.SourceCandidateCounters,
    ):
        counters.processed_tables = 3
        yield from candidate_tables

    def fake_evaluate(
        tables: list[dict[str, object]],
        context: builder.CandidateEvaluationContext,
        _args: object,
    ) -> list[builder.CandidateEvaluation]:
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
        "unrecoverable_replacement_rounds": 1,
        "unrecoverable_drop_probability": 1.0,
    }


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

    def fake_ensure_assets(
        entity_ids: set[str],
        context: builder.CandidateEvaluationContext,
        _args: object,
        _imageinfo_keys_accessed: set[str],
    ) -> None:
        for entity_id in entity_ids:
            asset_id = f"asset-{entity_id}"
            context.assets.setdefault(
                asset_id,
                {
                    "asset_id": asset_id,
                    "entity_id": entity_id,
                    "entity_wiki_title": context.entity_records[entity_id]["wiki_title"],
                    "asset_type": "text",
                },
            )
            context.entity_to_assets.setdefault(entity_id, [asset_id])

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

    monkeypatch.setattr(builder, "_ensure_candidate_assets", fake_ensure_assets)
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
            "--no_wikipedia",
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

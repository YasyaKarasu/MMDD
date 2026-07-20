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

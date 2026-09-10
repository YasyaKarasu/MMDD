from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts_old"
sys.path.insert(0, str(SCRIPTS))

from wdc200k_io import AtomicJsonlShard, StageFingerprint, StageManifest
from wdc200k_sampling import (
    SamplingPolicy,
    iter_sampled_records,
    sample_structural_artifacts,
    sample_table_entities,
    validate_sampling_source_authority,
    validate_sampling_artifacts,
    validate_sampling_consumed_paths,
)
import build_wdc200k_mm_joinability_dataset as pipeline
import build_mm_joinability_dataset as join_builder
import wdc200k_sampling as sampling_module
from stage1_io import stable_hash
from wdc200k_structural import derive_structural_evidence


def _entity_id(row_id: int) -> str:
    return f"ent_{stable_hash(f'wiki-{row_id}', length=16)}"


def _source(
    *,
    table_id: str = "table-1",
    rows: int = 8,
    blank_entities: set[int] | None = None,
    blank_values: set[int] | None = None,
) -> dict[str, Any]:
    blank_entities = blank_entities or set()
    blank_values = blank_values or set()
    records = []
    for row_id in range(rows):
        values = (
            "" if row_id in blank_values else f"value-{row_id}"
        )
        entity = "" if row_id in blank_entities else f"entity-{row_id}"
        records.append(
            {
                "row_id": row_id,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "Name",
                        "text": entity,
                        "wiki_title": f"wiki-{row_id}" if entity else "",
                    },
                    {"column_index": 1, "column_name": "Value", "text": values},
                    {"column_index": 2, "column_name": "Context", "text": values},
                ],
            }
        )
    return {
        "source_table_id": table_id,
        "num_rows": rows,
        "num_cols": 3,
        "columns": [
            {"column_index": 0, "column_name": "Name"},
            {"column_index": 1, "column_name": "Value"},
            {"column_index": 2, "column_name": "Context"},
        ],
        "rows": records,
        "metadata": {
            "candidate_entity_columns": [0],
            "column_profiles": [
                {"column_index": 0, "column_name": "Name", "non_empty_ratio": 1.0},
                {
                    "column_index": 1,
                    "column_name": "Value",
                    "non_empty_ratio": (rows - len(blank_values)) / rows,
                },
                {
                    "column_index": 2,
                    "column_name": "Context",
                    "non_empty_ratio": (rows - len(blank_values)) / rows,
                },
            ],
        },
    }


def _entities(table_id: str = "table-1", rows: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": _entity_id(row_id),
            "wiki_title": f"wiki-{row_id}",
            "appears_in": [
                {
                    "source_table_id": table_id,
                    "row_id": row_id,
                    "column_index": 0,
                    "column_name": "Name",
                }
            ],
            "page_url": "",
            "image_urls": [],
        }
        for row_id in range(rows)
    ]


def _page(entity_id: str, table_id: str = "table-1") -> dict[str, Any]:
    row_id = int(entity_id.rsplit("-", 1)[1])
    return {
        "url_key": f"page-{row_id}",
        "page_url": f"https://example.test/{entity_id}",
        "entity_id": _entity_id(row_id),
        "source_table_id": table_id,
        "row_id": row_id,
    }


def _image(entity_id: str, table_id: str = "table-1") -> dict[str, Any]:
    row_id = int(entity_id.rsplit("-", 1)[1])
    return {
        "url_key": f"image-{row_id}",
        "image_url": f"https://images.test/{entity_id}.jpg",
        "entity_id": _entity_id(row_id),
        "source_table_id": table_id,
        "row_id": row_id,
        "ordinal": 0,
    }


def _sample(
    source: dict[str, Any],
    *,
    pages: list[dict[str, Any]],
    images: list[dict[str, Any]],
    seed: int = 20260720,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return sample_table_entities(
        source,
        _entities(str(source["source_table_id"]), int(source["num_rows"])),
        pages,
        images,
        SamplingPolicy(entity_sampling_seed=seed),
    )


def test_prefilter_rejects_fewer_than_five_non_empty_entity_rows() -> None:
    source = _source(blank_entities={0, 1, 2, 3})
    pages = [_page(f"entity-{index}") for index in range(8)]

    sampled, decision = _sample(source, pages=pages, images=[])

    assert sampled == []
    assert decision["eligible"] is False
    assert decision["reason"] == "fewer_than_query_rows_non_empty_entities"


def test_prefilter_rejects_fewer_than_three_material_rows_for_one_attribute() -> None:
    source = _source(blank_values={4, 5, 6, 7})
    pages = [_page(f"entity-{index}") for index in range(2)]

    sampled, decision = _sample(source, pages=pages, images=[])

    assert sampled == []
    assert decision["eligible"] is False
    assert decision["reason"] == "insufficient_material_support_for_candidate"


def test_page_only_and_direct_only_entities_are_sampled() -> None:
    pages = [_page(f"entity-{index}") for index in (0, 1, 2, 6)]
    images = [_image(f"entity-{index}") for index in (3, 4, 5, 7)]

    sampled, decision = _sample(_source(), pages=pages, images=images)

    assert decision["eligible"] is True
    assert {record["stratum"] for record in sampled} == {
        "page_only",
        "direct_only",
    }
    assert len(sampled) == 8


def test_stratified_fill_mixes_all_available_url_types() -> None:
    pages = [_page(f"entity-{index}") for index in (0, 1, 2, 3, 4, 5)]
    images = [_image(f"entity-{index}") for index in (0, 1, 2, 6, 7)]

    sampled, _decision = _sample(_source(), pages=pages, images=images)

    assert {record["stratum"] for record in sampled} == {
        "both",
        "page_only",
        "direct_only",
    }
    assert [record["sampling_rank"] for record in sampled] == list(range(1, 9))
    assert sum(record["selection_reason"] == "anchor_support" for record in sampled) == 3
    assert all(record["supporting_candidate_attributes"] for record in sampled)


def test_sampling_is_stable_across_input_order_and_seed_controls_order() -> None:
    source = _source(rows=10)
    pages = [_page(f"entity-{index}") for index in range(10)]
    images = [_image(f"entity-{index}") for index in (0, 2, 4, 6, 8)]
    first, _ = sample_table_entities(
        source, _entities(rows=10), pages, images, SamplingPolicy(entity_sampling_seed=7)
    )
    reordered, _ = sample_table_entities(
        source,
        list(reversed(_entities(rows=10))),
        list(reversed(pages)),
        list(reversed(images)),
        SamplingPolicy(entity_sampling_seed=7),
    )
    other_seed, _ = sample_table_entities(
        source, _entities(rows=10), pages, images, SamplingPolicy(entity_sampling_seed=8)
    )

    assert [item["entity_id"] for item in first] == [item["entity_id"] for item in reordered]
    assert [item["entity_id"] for item in first] != [item["entity_id"] for item in other_seed]
    assert len(first) == 8
    assert len(source["rows"]) == 10


def test_per_table_limit_expands_only_named_table_and_preserves_prefix(
    tmp_path: Path,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _ = _structural_fixture(structural_root)
    initial = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "initial",
        policy=SamplingPolicy(sampled_entities_per_table=5),
    )
    expanded = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "expanded",
        policy=SamplingPolicy(sampled_entities_per_table=5),
        per_table_entity_limits={"table-1": 8},
    )

    initial_entities = list(iter_sampled_records(initial, "sampled_entities"))
    expanded_entities = list(iter_sampled_records(expanded, "sampled_entities"))
    assert len(initial_entities) == 5
    assert len(expanded_entities) == 8
    assert [item["entity_id"] for item in expanded_entities[:5]] == [
        item["entity_id"] for item in initial_entities
    ]
    decision = next(iter(iter_sampled_records(expanded, "prefilter_tables")))
    assert decision["sample_limit"] == 8


def test_sampling_progress_reports_real_shards_and_resume_completion(
    tmp_path: Path,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _ = _structural_fixture(structural_root)
    output_root = tmp_path / "sampling"
    updates: list[dict[str, Any]] = []

    sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
        progress_callback=updates.append,
    )

    assert [(item["completed"], item["total"]) for item in updates] == [
        (0, 1),
        (1, 1),
    ]
    resumed: list[dict[str, Any]] = []
    sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
        progress_callback=resumed.append,
    )
    assert resumed[-1] == {
        "phase": "sample_structural_shards",
        "completed": 1,
        "total": 1,
        "resumed": True,
    }


def test_page_only_sampling_is_stable_across_input_order() -> None:
    source = _source(rows=12)
    entities = _entities(rows=12)
    pages = [_page(f"entity-{index}") for index in range(12)]
    expected, _ = sample_table_entities(
        source, entities, pages, [], SamplingPolicy(entity_sampling_seed=19)
    )
    actual, _ = sample_table_entities(
        source,
        list(reversed(entities)),
        reversed(pages),
        [],
        SamplingPolicy(entity_sampling_seed=19),
    )
    assert [record["entity_id"] for record in actual] == [
        record["entity_id"] for record in expected
    ]


def test_default_five_row_query_requires_three_recoveries_for_one_attribute() -> None:
    common = {
        "valid_source_rows": set(range(5)),
        "query_rows_per_table": 5,
        "min_recovery_denominator": 2,
        "min_ratio": 0.6,
    }
    assert join_builder.recovery_column_profile(
        **common, recovered_source_rows={0, 1}
    ) is None
    profile = join_builder.recovery_column_profile(
        **common, recovered_source_rows={0, 1, 2}
    )
    assert profile is not None
    assert profile["required_recovered_rows"] == 3
    selected = join_builder.select_query_source_rows(
        source_row_order=list(range(5)),
        recovered_source_rows={0, 1, 2},
        query_rows_per_table=5,
        required_recovered_rows=profile["required_recovered_rows"],
    )
    assert len(selected) == 5
    assert set(selected) & {0, 1, 2} == {0, 1, 2}
    selected = join_builder.select_query_source_rows(
        source_row_order=list(range(8)),
        recovered_source_rows={0, 1, 2},
        query_rows_per_table=5,
        required_recovered_rows=3,
    )
    source = _source()
    selected_set = set(selected)
    query_rows, query_source_rows = join_builder.project_selected_rows(
        source, [0, 2], selected_set, min_required_cols=1
    )
    target_rows, target_source_rows = join_builder.project_selected_rows(
        source, [1, 2], selected_set, min_required_cols=0
    )
    assert len(query_rows) == len(target_rows) == 5
    assert query_source_rows == target_source_rows == selected


def _write_shard(root: Path, relative: str, records: list[dict[str, Any]]):
    writer = AtomicJsonlShard(root / relative)
    for record in records:
        writer.write(record)
    completed = writer.commit()
    return completed.__class__(relative, completed.records, completed.bytes, completed.sha256)


def _structural_fixture(root: Path) -> tuple[Path, dict[str, Any]]:
    source = _source()
    entities = _entities()
    pages = [_page(f"entity-{index}") for index in range(6)]
    images = [_image(f"entity-{index}") for index in (0, 1, 2, 6, 7)]
    completed = [
        _write_shard(root, "source_tables/part-00000.jsonl", [source]),
        _write_shard(root, "entities/part-00000.jsonl", entities),
        _write_shard(root, "page_refs/part-00000.jsonl", pages),
        _write_shard(root, "direct_image_refs/part-00000.jsonl", images),
        _write_shard(root, "structural_failures/part-00000.jsonl", []),
        _write_shard(
            root,
            "selection/validated-00000.jsonl",
            [{"source_table_id": "table-1", "rows": 8}],
        ),
    ]
    manifest = StageManifest(
        root / "structural-00000.json",
        StageFingerprint("wdc200k_structural", "input", "parameters", "wdc200k-structural-v2"),
    )
    for shard in completed:
        manifest.record_shard(shard)
    manifest.mark_complete()
    return manifest.path, source


def test_stage_preserves_source_and_downstream_iterators_expose_only_sampled(tmp_path: Path) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, source = _structural_fixture(structural_root)
    original_rows = json.loads(json.dumps(source["rows"]))

    result = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "sampling",
        policy=SamplingPolicy(sampled_entities_per_table=5),
    )

    sampled_entities = list(iter_sampled_records(result, "sampled_entities"))
    sampled_pages = list(iter_sampled_records(result, "sampled_page_refs"))
    sampled_images = list(iter_sampled_records(result, "sampled_direct_image_refs"))
    assert len(sampled_entities) == 5
    selected_ids = {record["entity_id"] for record in sampled_entities}
    assert {record["entity_id"] for record in sampled_pages} <= selected_ids
    assert {record["entity_id"] for record in sampled_images} <= selected_ids
    source_after = next(iter(json.loads(line) for line in (structural_root / "source_tables/part-00000.jsonl").read_text().splitlines()))
    assert source_after["rows"] == original_rows
    assert result.eligible_tables == 1


def test_stage_resume_reuses_valid_shards_without_duplicate_writes(tmp_path: Path) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    output_root = tmp_path / "sampling"

    first = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
    )
    before = first.manifest_path.read_bytes()
    shard_bytes = {path: path.read_bytes() for paths in first.artifact_paths.values() for path in paths}
    second = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
    )

    assert second.manifest_path.read_bytes() == before
    assert {path: path.read_bytes() for paths in second.artifact_paths.values() for path in paths} == shard_bytes
    assert len(list(iter_sampled_records(second, "sampled_entities"))) == 8


def test_sampling_reconstructs_compacted_evidence_from_source_tables(
    tmp_path: Path,
) -> None:
    structural_root = tmp_path / "structural"
    source = _source()
    source["columns"].extend(
        [
            {"column_index": 3, "column_name": "page_url"},
            {"column_index": 4, "column_name": "image"},
        ]
    )
    source["num_cols"] = 5
    for row in source["rows"]:
        row_id = int(row["row_id"])
        page_url = f"https://example.test/entity-{row_id}"
        image_url = f"https://images.test/entity-{row_id}.jpg"
        row["cells"].extend(
            [
                {
                    "column_index": 3,
                    "column_name": "page_url",
                    "raw": page_url,
                    "text": page_url,
                },
                {
                    "column_index": 4,
                    "column_name": "image",
                    "raw": image_url,
                    "text": image_url,
                },
            ]
        )
    entities, pages, images = derive_structural_evidence(source)
    completed = [
        _write_shard(
            structural_root,
            "source_tables/part-00000.jsonl",
            [source],
        ),
        _write_shard(
            structural_root,
            "entities/part-00000.jsonl",
            entities,
        ),
        _write_shard(
            structural_root,
            "page_refs/part-00000.jsonl",
            pages,
        ),
        _write_shard(
            structural_root,
            "direct_image_refs/part-00000.jsonl",
            images,
        ),
        _write_shard(
            structural_root,
            "structural_failures/part-00000.jsonl",
            [],
        ),
        _write_shard(
            structural_root,
            "selection/validated-00000.jsonl",
            [{"source_table_id": "table-1", "rows": 8}],
        ),
    ]
    structural_manifest = StageManifest(
        structural_root / "structural-00000.json",
        StageFingerprint(
            "wdc200k_structural",
            "input",
            "parameters",
            "wdc200k-structural-v2",
        ),
    )
    for shard in completed:
        structural_manifest.record_shard(shard)
    structural_manifest.mark_complete()

    baseline = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[structural_manifest.path],
        output_root=tmp_path / "baseline",
        policy=SamplingPolicy(),
    )
    for relative in (
        "entities/part-00000.jsonl",
        "page_refs/part-00000.jsonl",
        "direct_image_refs/part-00000.jsonl",
        "selection/validated-00000.jsonl",
    ):
        (structural_root / relative).unlink()
    compact = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[structural_manifest.path],
        output_root=tmp_path / "compact",
        policy=SamplingPolicy(),
    )

    assert {
        artifact: [path.read_bytes() for path in paths]
        for artifact, paths in compact.artifact_paths.items()
    } == {
        artifact: [path.read_bytes() for path in paths]
        for artifact, paths in baseline.artifact_paths.items()
    }


def test_completed_resume_does_not_repeat_full_source_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _ = _structural_fixture(structural_root)
    output_root = tmp_path / "sampling"
    sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
    )
    import wdc200k_sampling as sampling_module

    monkeypatch.setattr(
        sampling_module,
        "_validate_sampling_closure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume repeated full closure")
        ),
    )

    resumed = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
    )
    assert resumed.sampled_entities == 8


def test_incomplete_manifest_resumes_missing_artifact_without_rewriting_completed(
    tmp_path: Path,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    output_root = tmp_path / "sampling"
    first = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
    )
    payload = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    missing = next(
        item
        for item in payload["completed_shards"]
        if item["path"].startswith("sampled_page_refs/")
    )
    payload["completed_shards"].remove(missing)
    payload["complete"] = False
    payload["totals"] = {
        "shards": len(payload["completed_shards"]),
        "records": sum(item["records"] for item in payload["completed_shards"]),
        "bytes": sum(item["bytes"] for item in payload["completed_shards"]),
    }
    first.manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / missing["path"]).unlink()
    retained = {
        path: path.read_bytes()
        for artifact, paths in first.artifact_paths.items()
        if artifact != "sampled_page_refs"
        for path in paths
    }

    resumed = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
    )

    assert all(path.read_bytes() == content for path, content in retained.items())
    assert resumed.sampled_entities == 8
    assert resumed.sampled_page_refs == 6


def test_real_commit_interruption_resumes_with_identical_bytes_and_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_structural = tmp_path / "baseline-structural"
    baseline_manifest, _ = _structural_fixture(baseline_structural)
    baseline = sample_structural_artifacts(
        structural_output_root=baseline_structural,
        structural_manifests=[baseline_manifest],
        output_root=tmp_path / "baseline-sampling",
        policy=SamplingPolicy(),
    )
    expected = {
        artifact: [path.read_bytes() for path in paths]
        for artifact, paths in baseline.artifact_paths.items()
    }

    structural_root = tmp_path / "interrupted-structural"
    manifest_path, _ = _structural_fixture(structural_root)
    original_commit = AtomicJsonlShard.commit
    interrupted = False

    def interrupt_page_commit(self):
        nonlocal interrupted
        if not interrupted and self.path.parent.name == "sampled_page_refs":
            interrupted = True
            raise RuntimeError("commit interruption")
        return original_commit(self)

    monkeypatch.setattr(AtomicJsonlShard, "commit", interrupt_page_commit)
    with pytest.raises(RuntimeError, match="commit interruption"):
        sample_structural_artifacts(
            structural_output_root=structural_root,
            structural_manifests=[manifest_path],
            output_root=tmp_path / "interrupted-sampling",
            policy=SamplingPolicy(),
        )
    monkeypatch.setattr(AtomicJsonlShard, "commit", original_commit)
    resumed = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "interrupted-sampling",
        policy=SamplingPolicy(),
    )

    assert {
        artifact: [path.read_bytes() for path in paths]
        for artifact, paths in resumed.artifact_paths.items()
    } == expected


def test_authority_publish_interruption_leaves_manifest_incomplete_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _ = _structural_fixture(structural_root)
    output_root = tmp_path / "sampling"
    import wdc200k_sampling as sampling_module

    original_publish = sampling_module._publish_compact_authority

    def interrupt(*_args, **_kwargs):
        raise RuntimeError("authority publish interrupted")

    monkeypatch.setattr(sampling_module, "_publish_compact_authority", interrupt)
    with pytest.raises(RuntimeError, match="authority publish"):
        sample_structural_artifacts(
            structural_output_root=structural_root,
            structural_manifests=[manifest_path],
            output_root=output_root,
            policy=SamplingPolicy(),
        )
    assert json.loads((output_root / "manifest.json").read_text())["complete"] is False

    monkeypatch.setattr(
        sampling_module,
        "_publish_compact_authority",
        original_publish,
    )
    resumed = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
    )
    assert resumed.sampled_entities == 8


def test_sampling_policy_defaults_and_rejects_budget_that_cannot_cover_tables() -> None:
    policy = SamplingPolicy()
    assert policy.sampled_entities_per_table == 8
    assert policy.query_rows_per_table == 5
    assert policy.min_column_non_empty_ratio == 0.5
    assert policy.min_recovered_value_ratio == 0.6
    assert policy.min_recovery_denominator == 2
    assert policy.min_rows_per_output_table == 2
    with pytest.raises(ValueError, match="global entity budget"):
        SamplingPolicy(global_entity_budget=15).validate_budget(eligible_tables=2)


def test_prefilter_uses_canonical_entity_column_choice() -> None:
    source = _source(rows=5)
    source["metadata"]["candidate_entity_columns"] = [1, 0]
    source["metadata"]["column_profiles"][0]["wiki_link_ratio"] = 1.0
    source["metadata"]["column_profiles"][1]["wiki_link_ratio"] = 0.0
    source["metadata"]["column_profiles"][1]["non_empty_ratio"] = 0.0
    for row in source["rows"]:
        row["cells"][1]["text"] = ""
    pages = [_page(f"entity-{index}") for index in range(5)]

    sampled, decision = _sample(source, pages=pages, images=[])

    assert decision["eligible"] is True
    assert len(sampled) == 5


def test_prefilter_rejects_urls_rejected_by_structural_normalizer() -> None:
    malformed = [
        {
            **_page(f"entity-{index}"),
            "page_url": "https://[",
        }
        for index in range(5)
    ]

    sampled, decision = _sample(_source(rows=5), pages=malformed, images=[])

    assert sampled == []
    assert decision["reason"] == "insufficient_material_support_for_candidate"


def test_global_budget_uses_actual_sampled_count(tmp_path: Path) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)

    result = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "sampling",
        policy=SamplingPolicy(
            sampled_entities_per_table=10,
            global_entity_budget=8,
        ),
    )

    assert result.sampled_entities == 8


def test_insufficient_global_budget_reports_shortfall_and_stays_incomplete(
    tmp_path: Path,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    output_root = tmp_path / "sampling"

    with pytest.raises(
        ValueError,
        match=r"budget=7, required=8, shortfall=1",
    ):
        sample_structural_artifacts(
            structural_output_root=structural_root,
            structural_manifests=[manifest_path],
            output_root=output_root,
            policy=SamplingPolicy(global_entity_budget=7),
        )

    payload = json.loads((output_root / "manifest.json").read_text())
    assert payload["complete"] is False


def test_completed_sampling_rejects_cross_artifact_closure_violation(
    tmp_path: Path,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    output_root = tmp_path / "sampling"
    result = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=output_root,
        policy=SamplingPolicy(),
    )
    page_path = result.artifact_paths["sampled_page_refs"][0]
    records = [json.loads(line) for line in page_path.read_text().splitlines()]
    records[0]["entity_id"] = "not-sampled"
    writer = AtomicJsonlShard(page_path)
    for record in records:
        writer.write(record)
    completed = writer.commit()
    payload = json.loads(result.manifest_path.read_text())
    declared = next(
        item
        for item in payload["completed_shards"]
        if item["path"].startswith("sampled_page_refs/")
    )
    declared.update(
        records=completed.records,
        bytes=completed.bytes,
        sha256=completed.sha256,
    )
    payload["totals"] = {
        "shards": len(payload["completed_shards"]),
        "records": sum(item["records"] for item in payload["completed_shards"]),
        "bytes": sum(item["bytes"] for item in payload["completed_shards"]),
    }
    result.manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(ValueError, match="closure"):
        sample_structural_artifacts(
            structural_output_root=structural_root,
            structural_manifests=[manifest_path],
            output_root=output_root,
            policy=SamplingPolicy(),
        )


def test_compact_authority_survives_removed_full_entity_and_reference_shards(
    tmp_path: Path,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    result = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "sampling",
        policy=SamplingPolicy(),
    )
    for directory in ("entities", "page_refs", "direct_image_refs"):
        for path in (structural_root / directory).glob("*.jsonl"):
            path.unlink()

    authority = validate_sampling_source_authority(
        result.manifest_path,
        structural_output_root=structural_root,
    )

    assert authority.source_tables == (
        structural_root / "source_tables/part-00000.jsonl",
    )
    assert authority.source_tables_count == 1


def test_sampling_source_authority_reuses_unchanged_process_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    result = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "sampling",
        policy=SamplingPolicy(),
    )
    sampling_module._SOURCE_AUTHORITY_VALIDATION_CACHE.clear()
    original_validate = sampling_module.validate_completed_shard
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        sampling_module,
        "validate_completed_shard",
        counted,
    )
    first = validate_sampling_source_authority(
        result.manifest_path,
        structural_output_root=structural_root,
    )
    first_calls = calls
    second = validate_sampling_source_authority(
        result.manifest_path,
        structural_output_root=structural_root,
    )

    assert first == second
    assert first_calls > 0
    assert calls == first_calls

    source_path = first.source_tables[0]
    source_stat = source_path.stat()
    os.utime(
        source_path,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000),
    )
    validate_sampling_source_authority(
        result.manifest_path,
        structural_output_root=structural_root,
    )
    assert calls > first_calls


def test_sampling_authority_rejects_unknown_completed_shard(tmp_path: Path) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    result = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "sampling",
        policy=SamplingPolicy(),
    )
    extra = _write_shard(result.output_root, "unknown/part-00000.jsonl", [])
    payload = json.loads(result.manifest_path.read_text())
    payload["completed_shards"].append(
        {
            "path": "unknown/part-00000.jsonl",
            "records": extra.records,
            "bytes": extra.bytes,
            "sha256": extra.sha256,
        }
    )
    result.manifest_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="artifact.*set"):
        validate_sampling_artifacts(result.manifest_path)


def test_sampling_closure_rejects_resigned_wrong_entity_row_and_wiki(
    tmp_path: Path,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    result = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "sampling",
        policy=SamplingPolicy(),
    )
    entity_path = result.artifact_paths["sampled_entities"][0]
    records = [json.loads(line) for line in entity_path.read_text().splitlines()]
    records[0]["source_row_id"] = 999
    records[0]["wiki_title"] = "wrong-wiki"
    writer = AtomicJsonlShard(entity_path)
    for record in records:
        writer.write(record)
    completed = writer.commit()
    payload = json.loads(result.manifest_path.read_text())
    declared = next(
        item
        for item in payload["completed_shards"]
        if item["path"].startswith("sampled_entities/")
    )
    declared.update(
        records=completed.records,
        bytes=completed.bytes,
        sha256=completed.sha256,
    )
    payload["totals"] = {
        "shards": len(payload["completed_shards"]),
        "records": sum(item["records"] for item in payload["completed_shards"]),
        "bytes": sum(item["bytes"] for item in payload["completed_shards"]),
    }
    result.manifest_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="closure"):
        sample_structural_artifacts(
            structural_output_root=structural_root,
            structural_manifests=[manifest_path],
            output_root=result.output_root,
            policy=SamplingPolicy(),
        )


def test_sampling_closure_rejects_resigned_wrong_entity_id(tmp_path: Path) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    result = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "sampling",
        policy=SamplingPolicy(),
    )
    entity_path = result.artifact_paths["sampled_entities"][0]
    records = [json.loads(line) for line in entity_path.read_text().splitlines()]
    records[0]["entity_id"] = "ent_resigned_but_wrong"
    writer = AtomicJsonlShard(entity_path)
    for record in records:
        writer.write(record)
    completed = writer.commit()
    payload = json.loads(result.manifest_path.read_text())
    declared = next(
        item
        for item in payload["completed_shards"]
        if item["path"].startswith("sampled_entities/")
    )
    declared.update(
        records=completed.records,
        bytes=completed.bytes,
        sha256=completed.sha256,
    )
    payload["totals"] = {
        "shards": len(payload["completed_shards"]),
        "records": sum(item["records"] for item in payload["completed_shards"]),
        "bytes": sum(item["bytes"] for item in payload["completed_shards"]),
    }
    result.manifest_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="closure"):
        sample_structural_artifacts(
            structural_output_root=structural_root,
            structural_manifests=[manifest_path],
            output_root=result.output_root,
            policy=SamplingPolicy(),
        )


def test_sampling_consumers_reject_poison_entity_and_page_paths(
    tmp_path: Path,
) -> None:
    structural_root = tmp_path / "structural"
    manifest_path, _source_record = _structural_fixture(structural_root)
    result = sample_structural_artifacts(
        structural_output_root=structural_root,
        structural_manifests=[manifest_path],
        output_root=tmp_path / "sampling",
        policy=SamplingPolicy(),
    )
    supplied = dict(result.artifact_paths)
    for artifact in ("sampled_entities", "sampled_page_refs"):
        poison = tmp_path / f"poison-{artifact}.jsonl"
        poison.write_text(supplied[artifact][0].read_text(encoding="utf-8"))
        poisoned = {**supplied, artifact: (poison,)}
        with pytest.raises(ValueError, match="manifest authority"):
            validate_sampling_consumed_paths(result.manifest_path, poisoned)


def test_pipeline_cli_exposes_sampling_defaults(tmp_path: Path) -> None:
    args = pipeline.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
        ]
    )

    assert args.sampled_entities_per_table == 8
    assert args.entity_sampling_seed == 20260720
    assert args.global_entity_budget is None
    assert "sampling" in pipeline.STAGES

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.construction import (
    DEFAULT_MAX_CELL_CHARS,
    _artifact_records,
    build_stage1_training_artifacts,
    serialize_table_parts,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _table(table_id: str, headers: list[str], values: list[list[str]]) -> dict:
    return {
        "table_id": table_id,
        "split": "train",
        "source_table_id": f"source_{table_id}",
        "columns": [
            {"column_index": index, "column_name": header}
            for index, header in enumerate(headers)
        ],
        "rows": [
            {
                "row_id": row_index,
                "cells": [
                    {"column_index": column_index, "text": value}
                    for column_index, value in enumerate(row)
                ],
            }
            for row_index, row in enumerate(values)
        ],
    }


def test_serialize_table_parts_limits_cleaned_cell_text():
    long_value = "  prefix   " + "x" * DEFAULT_MAX_CELL_CHARS
    parts = serialize_table_parts(
        _table("q", ["Entity", "Value"], [[long_value, "short"]]),
        max_rows=1,
    )

    assert parts == [
        "Columns: Entity | Value",
        "Row: prefix " + "x" * (DEFAULT_MAX_CELL_CHARS - len("prefix ")) + " | short",
    ]


def test_serialize_table_parts_can_repeat_column_names():
    parts = serialize_table_parts(
        _table("q", ["Entity", "", "Value"], [["Messi", "Argentina", "10"]]),
        max_rows=1,
        row_format="named_cells",
    )

    assert parts == [
        "Columns: Entity |  | Value",
        "Row: Entity: Messi | column_1: Argentina | Value: 10",
    ]


def test_artifact_records_prefers_manifest_over_stale_flat_file(tmp_path: Path):
    stale = {"table_id": "stale"}
    listed = {"table_id": "listed"}
    _write_jsonl(tmp_path / "query_tables.jsonl", [stale])
    shard = tmp_path / "query_tables" / "part-00000.jsonl"
    shard.parent.mkdir()
    _write_jsonl(shard, [listed])
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "format": "sharded_jsonl",
                "artifacts": {
                    "query_tables": {
                        "shards": [
                            {
                                "path": "query_tables/part-00000.jsonl",
                                "records": 1,
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert _artifact_records(tmp_path, "query_tables") == [listed]


def test_recovery_only_dataset_does_not_promote_source_assets_to_query_positives(tmp_path):
    records = {
        "query_tables": [_table("implicit", ["Title"], [["Book A"]]),
                         _table("explicit", ["Author"], [["Jane Doe"]])],
        "data_lake_tables": [_table("positive", ["Author"], [["Jane Doe"]]),
                             _table("negative", ["Author"], [["Alex Smith"]])],
        "bridge_assets": [{"asset_id": "reviewed", "asset_type": "text", "content": "Jane Doe",
                           "source_table_id": "source_positive"},
                          {"asset_id": "unreviewed", "asset_type": "text", "content": "Shipping policy",
                           "source_table_id": "source_positive"}],
        "qrels": [{"query_table_id": q, "target_table_id": "positive", "rel": 3}
                  for q in ("implicit", "explicit")],
        "evidence_recoveries": [{"query_table_id": "implicit", "target_table_id": "positive",
                                 "query_row_id": 0, "evidence": {"asset_id": "reviewed"}}],
    }
    manifest = {"artifacts": {}, "curation": {
        "evidence_supervision": "recovery_records_only_no_provenance_fallback"}}
    for name, values in records.items():
        _write_jsonl(tmp_path / f"{name}.jsonl", values)
        manifest["artifacts"][name] = {"shards": [{"path": f"{name}.jsonl"}]}
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest))
    artifacts = build_stage1_training_artifacts(tmp_path, dataset_name="strict")
    explicit = next(r for r in artifacts["target_lists"] if r["query_id"] == "explicit")
    assert explicit["positive_evidence_by_target"] == {"positive": []}
    assert not explicit["has_recovery_supervision"]
    assert not any(r["query_id"] == "explicit" and r["destination_type"] != "table" for r in artifacts["edge_lists"])
    assert not any(r["positive_id"] == "unreviewed" for r in artifacts["edge_lists"])
    assert "unreviewed" in {r["object_id"] for r in artifacts["stage1_corpus"]}


def test_stage1_constructor_builds_all_files_and_four_initial_negative_kinds(tmp_path):
    query = _table("q", ["Player", "Country"], [["Messi", "Argentina"], ["Mbappe", "France"]])
    query.update(
        page_title="SECRET_PAGE_TITLE",
        caption="SECRET_CAPTION",
        section_title="SECRET_SECTION",
    )
    targets = [
        _table("positive", ["Country", "Club"], [["Argentina", "Barcelona"]]),
        _table("semantic", ["Player", "Country"], [["Messi", "Spain"]]),
        _table("structure", ["Code", "Value"], [["A", "1"], ["B", "2"]]),
        _table("corrupted", ["Name"], [["A"]]),
        _table("random", ["Year", "Score", "Rank"], [["2020", "4", "1"]]),
    ]
    assets = [
        {"asset_id": f"e_{target['table_id']}", "asset_type": "text", "content": target["table_id"]}
        for target in targets
    ]
    image_path = tmp_path / "e_image.bin"
    image_path.write_bytes(b"image")
    assets.append(
        {
            "asset_id": "e_image",
            "asset_type": "image",
            "local_path": str(image_path),
            "entity_wiki_title": "SECRET_IMAGE_LABEL",
            "source": "SECRET_IMAGE_SOURCE",
        }
    )
    recoveries = [
        {
            "target_table_id": target["table_id"],
            "evidence": {"asset_id": f"e_{target['table_id']}"},
        }
        for target in targets
    ]
    _write_jsonl(tmp_path / "query_tables.jsonl", [query])
    _write_jsonl(tmp_path / "data_lake_tables.jsonl", targets)
    _write_jsonl(tmp_path / "bridge_assets.jsonl", assets)
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_table_id": "q", "target_table_id": "positive", "split": "train"}],
    )
    _write_jsonl(tmp_path / "evidence_recoveries.jsonl", recoveries)

    artifacts = build_stage1_training_artifacts(
        tmp_path,
        dataset_name="synthetic",
        max_rows=2,
        seed=7,
    )

    assert set(artifacts) == {"stage1_objects", "edge_lists", "target_lists", "stage1_corpus"}
    table_object = next(record for record in artifacts["stage1_objects"] if record["object_id"] == "q")
    assert table_object["embedding_role"] == "query"
    assert set(table_object) == {"object_id", "object_type", "embedding_role", "table_parts"}
    assert "SECRET_" not in "\n".join(table_object["table_parts"])
    target_object = next(
        record for record in artifacts["stage1_objects"] if record["object_id"] == "positive"
    )
    assert target_object["embedding_role"] == "target"
    evidence_object = next(
        record for record in artifacts["stage1_objects"] if record["object_id"] == "e_positive"
    )
    assert set(evidence_object) == {"object_id", "object_type", "text"}
    image_object = next(
        record for record in artifacts["stage1_objects"] if record["object_id"] == "e_image"
    )
    assert "text" not in image_object
    candidates = artifacts["target_lists"][0]["candidates"]
    assert artifacts["target_lists"][0]["direct_positive_target_id"] == "positive"
    assert artifacts["target_lists"][0]["evidence_positive_target_id"] == "positive"
    assert all(set(candidate) == {"target_id", "evidence_ids"} for candidate in candidates)
    by_target = {candidate["target_id"]: candidate for candidate in candidates}
    assert by_target["corrupted"]["evidence_ids"] == ["e_positive"]
    edge = artifacts["edge_lists"][0]
    assert edge["destination_type"] == "table"
    assert len(edge["candidate_ids"]) == 5
    corpus_ids = {record["object_id"] for record in artifacts["stage1_corpus"]}
    assert all(set(record) == {"object_id"} for record in artifacts["stage1_corpus"])
    assert "q" not in corpus_ids
    assert {"positive", "e_positive"} <= corpus_ids


def test_stage1_constructor_builds_cross_modal_edge_lists(tmp_path):
    query = _table("q", ["Entity"], [["A"]])
    positive = _table("positive", ["Value"], [["1"]])
    negative = _table("negative", ["Value"], [["2"]])
    positive_image = tmp_path / "positive.bin"
    negative_image = tmp_path / "negative.bin"
    positive_image.write_bytes(b"positive")
    negative_image.write_bytes(b"negative")
    assets = [
        {"asset_id": "positive_text", "asset_type": "text", "content": "positive"},
        {
            "asset_id": "positive_text_extra",
            "asset_type": "text",
            "content": "positive extra",
        },
        {
            "asset_id": "positive_image",
            "asset_type": "image",
            "local_path": str(positive_image),
        },
        {"asset_id": "negative_text", "asset_type": "text", "content": "negative"},
        {
            "asset_id": "negative_text_extra",
            "asset_type": "text",
            "content": "negative extra",
        },
        {
            "asset_id": "negative_image",
            "asset_type": "image",
            "local_path": str(negative_image),
        },
    ]
    recoveries = [
        {"target_table_id": "positive", "evidence": {"asset_id": "positive_text"}},
        {
            "target_table_id": "positive",
            "evidence": {"asset_id": "positive_text_extra"},
        },
        {"target_table_id": "positive", "evidence": {"asset_id": "positive_image"}},
        {"target_table_id": "negative", "evidence": {"asset_id": "negative_text"}},
        {
            "target_table_id": "negative",
            "evidence": {"asset_id": "negative_text_extra"},
        },
        {"target_table_id": "negative", "evidence": {"asset_id": "negative_image"}},
    ]
    _write_jsonl(tmp_path / "query_tables.jsonl", [query])
    _write_jsonl(tmp_path / "data_lake_tables.jsonl", [positive, negative])
    _write_jsonl(tmp_path / "bridge_assets.jsonl", assets)
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_table_id": "q", "target_table_id": "positive", "split": "train"}],
    )
    _write_jsonl(tmp_path / "evidence_recoveries.jsonl", recoveries)

    artifacts = build_stage1_training_artifacts(tmp_path, dataset_name="synthetic")

    candidates = {
        candidate["target_id"]: candidate
        for candidate in artifacts["target_lists"][0]["candidates"]
    }
    assert candidates["positive"]["evidence_ids"] == [
        "positive_text",
        "positive_text_extra",
        "positive_image",
    ]
    edges = artifacts["edge_lists"]
    assert {
        (record["source_type"], record["destination_type"])
        for record in edges
    } == {
        ("table", "table"),
        ("table", "text"),
        ("text", "table"),
        ("table", "image"),
        ("image", "table"),
    }
    assert [
        record["positive_id"]
        for record in edges
        if record["query_id"] == "q" and record["destination_type"] == "text"
    ] == ["positive_text", "positive_text_extra"]
    assert {
        record["query_id"]
        for record in edges
        if record["destination_type"] == "table"
        and record["source_type"] in {"text", "image"}
    } == {"positive_text", "positive_text_extra", "positive_image"}


def test_stage1_constructor_uses_path_only_target_as_direct_hard_negative(tmp_path):
    query = _table("q", ["Entity"], [["A"]])
    direct = _table("direct", ["Key"], [["A"]])
    path_only = _table("path_only", ["Value"], [["1"]])
    negative = _table("negative", ["Other"], [["2"]])
    assets = [
        {"asset_id": "path_evidence", "asset_type": "text", "content": "A has value 1"},
        {"asset_id": "direct_evidence", "asset_type": "text", "content": "direct context"},
    ]
    recoveries = [
        {
            "query_table_id": "q",
            "target_table_id": "path_only",
            "evidence": {"asset_id": "path_evidence"},
        },
        {"target_table_id": "direct", "evidence": {"asset_id": "direct_evidence"}},
    ]
    _write_jsonl(tmp_path / "query_tables.jsonl", [query])
    _write_jsonl(tmp_path / "data_lake_tables.jsonl", [direct, path_only, negative])
    _write_jsonl(tmp_path / "bridge_assets.jsonl", assets)
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_table_id": "q", "target_table_id": "direct", "split": "train"}],
    )
    _write_jsonl(tmp_path / "evidence_recoveries.jsonl", recoveries)

    artifacts = build_stage1_training_artifacts(tmp_path, dataset_name="synthetic")

    target_list = artifacts["target_lists"][0]
    assert target_list["direct_positive_target_id"] == "direct"
    assert target_list["evidence_positive_target_id"] == "path_only"
    assert target_list["positive_target_ids"] == ["direct", "path_only"]
    direct_edge = next(
        edge
        for edge in artifacts["edge_lists"]
        if edge["query_id"] == "q" and edge["destination_type"] == "table"
    )
    assert direct_edge["positive_id"] == "direct"
    assert direct_edge["candidate_ids"][1] == "path_only"
    evidence_target_edge = next(
        edge
        for edge in artifacts["edge_lists"]
        if edge["query_id"] == "path_evidence" and edge["destination_type"] == "table"
    )
    assert evidence_target_edge["positive_id"] == "path_only"
    assert "direct" in evidence_target_edge["candidate_ids"]


def test_stage1_constructor_candidates_cover_all_multi_positive_targets(tmp_path):
    query = _table("q", ["Entity"], [["A"]])
    positive_a = _table("positive_a", ["Key"], [["A"]])
    positive_b = _table("positive_b", ["Value"], [["1"]])
    negative = _table("negative", ["Other"], [["2"]])
    _write_jsonl(tmp_path / "query_tables.jsonl", [query])
    _write_jsonl(
        tmp_path / "data_lake_tables.jsonl",
        [positive_a, positive_b, negative],
    )
    _write_jsonl(
        tmp_path / "bridge_assets.jsonl",
        [{"asset_id": "unused", "asset_type": "text", "content": "unrelated"}],
    )
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [
            {"query_table_id": "q", "target_table_id": "positive_a", "split": "train"},
            {"query_table_id": "q", "target_table_id": "positive_b", "split": "train"},
        ],
    )

    artifacts = build_stage1_training_artifacts(tmp_path, dataset_name="synthetic")

    target_list = artifacts["target_lists"][0]
    assert target_list["positive_target_ids"] == ["positive_a", "positive_b"]
    assert [candidate["target_id"] for candidate in target_list["candidates"]][:2] == [
        "positive_a",
        "positive_b",
    ]


def test_stage1_constructor_keeps_direct_training_without_recovery(tmp_path):
    query = _table("q", ["Entity"], [["A"]])
    direct = _table("direct", ["Key"], [["A"]])
    negative = _table("negative", ["Key"], [["B"]])
    _write_jsonl(tmp_path / "query_tables.jsonl", [query])
    _write_jsonl(tmp_path / "data_lake_tables.jsonl", [direct, negative])
    _write_jsonl(
        tmp_path / "bridge_assets.jsonl",
        [{"asset_id": "unused", "asset_type": "text", "content": "unrelated"}],
    )
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_table_id": "q", "target_table_id": "direct", "split": "train"}],
    )

    artifacts = build_stage1_training_artifacts(tmp_path, dataset_name="synthetic")

    target_list = artifacts["target_lists"][0]
    assert target_list["direct_positive_target_id"] == "direct"
    assert target_list["candidates"][0]["evidence_ids"] == []
    direct_edge = next(
        edge
        for edge in artifacts["edge_lists"]
        if edge["query_id"] == "q" and edge["destination_type"] == "table"
    )
    assert direct_edge["positive_id"] == "direct"


def test_stage1_constructor_uses_one_shared_data_lake_for_dev_queries(tmp_path):
    query = _table("q", ["Entity"], [["A"]])
    query["split"] = "dev"
    positive = _table("positive", ["Value"], [["1"]])
    negative = _table("negative", ["Value"], [["2"]])
    positive.pop("split")
    negative.pop("split")
    _write_jsonl(tmp_path / "query_tables.jsonl", [query])
    _write_jsonl(tmp_path / "data_lake_tables.jsonl", [positive, negative])
    _write_jsonl(
        tmp_path / "bridge_assets.jsonl",
        [{"asset_id": "unused", "asset_type": "text", "content": "unrelated"}],
    )
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_table_id": "q", "target_table_id": "positive", "split": "dev"}],
    )

    artifacts = build_stage1_training_artifacts(tmp_path, dataset_name="synthetic")

    assert len(artifacts["target_lists"]) == 1
    target_list = artifacts["target_lists"][0]
    assert target_list["split"] == "dev"
    assert {candidate["target_id"] for candidate in target_list["candidates"]} == {
        "positive",
        "negative",
    }


def test_stage1_constructor_resolves_raw_data_lake_source_references(tmp_path):
    query = _table("q", ["Entity"], [["A"]])
    positive = _table("positive", ["Value"], [["1"]])
    source = _table("unused", ["Entity", "Other"], [["B", "2"]])
    source.pop("table_id")
    source["source_table_id"] = "source_raw"
    raw_target = {
        "table_id": "raw",
        "split": "train",
        "source_table_id": "source_raw",
        "source_table_ref": {"artifact": "source_tables", "source_table_id": "source_raw"},
    }
    _write_jsonl(tmp_path / "query_tables.jsonl", [query])
    _write_jsonl(tmp_path / "data_lake_tables.jsonl", [positive, raw_target])
    _write_jsonl(tmp_path / "source_tables.jsonl", [source])
    _write_jsonl(
        tmp_path / "bridge_assets.jsonl",
        [{"asset_id": "e", "asset_type": "text", "content": "evidence"}],
    )
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_table_id": "q", "target_table_id": "positive", "split": "train"}],
    )
    _write_jsonl(
        tmp_path / "evidence_recoveries.jsonl",
        [{"target_table_id": "positive", "evidence": {"asset_id": "e"}}],
    )

    artifacts = build_stage1_training_artifacts(tmp_path, dataset_name="synthetic")

    raw_object = next(record for record in artifacts["stage1_objects"] if record["object_id"] == "raw")
    assert raw_object["table_parts"][0] == "Columns: Entity | Other"

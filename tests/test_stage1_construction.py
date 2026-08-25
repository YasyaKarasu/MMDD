from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.construction import build_stage1_training_artifacts


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _table(table_id: str, headers: list[str], values: list[list[str]]) -> dict:
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
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
        max_evidence_per_target=1,
        seed=7,
    )

    assert set(artifacts) == {"stage1_objects", "edge_lists", "target_lists", "stage1_corpus"}
    table_object = next(record for record in artifacts["stage1_objects"] if record["object_id"] == "q")
    assert table_object["embedding_role"] == "query"
    assert table_object["text"] == "\n".join(table_object["table_parts"])
    assert "SECRET_" not in table_object["text"]
    assert table_object["row_routing_texts"] == [
        "Columns: Player | Country\nRow: Messi | Argentina",
        "Columns: Player | Country\nRow: Mbappe | France",
    ]
    target_object = next(
        record for record in artifacts["stage1_objects"] if record["object_id"] == "positive"
    )
    assert target_object["embedding_role"] == "target"
    assert "row_routing_texts" not in target_object
    evidence_object = next(
        record for record in artifacts["stage1_objects"] if record["object_id"] == "e_positive"
    )
    assert evidence_object["embedding_role"] == "evidence"
    image_object = next(
        record for record in artifacts["stage1_objects"] if record["object_id"] == "e_image"
    )
    assert "text" not in image_object
    candidates = artifacts["target_lists"][0]["candidates"]
    negatives = {candidate["negative_source"]: candidate for candidate in candidates[1:]}
    assert set(negatives) == {
        "random",
        "semantic_similar_non_joinable",
        "type_structure_matched",
        "corrupted_path",
    }
    assert negatives["corrupted_path"]["evidence_ids"] == ["e_positive"]
    assert all(len(candidate["evidence_ids"]) <= 1 for candidate in candidates)
    edge = artifacts["edge_lists"][0]
    assert edge["destination_type"] == "table"
    assert len(edge["candidate_ids"]) == 5
    corpus_ids = {record["object_id"] for record in artifacts["stage1_corpus"]}
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

    artifacts = build_stage1_training_artifacts(
        tmp_path,
        dataset_name="synthetic",
        max_evidence_per_target=2,
    )

    edges = {
        (record["source_type"], record["destination_type"]): record
        for record in artifacts["edge_lists"]
    }
    assert set(edges) == {
        ("table", "table"),
        ("table", "text"),
        ("text", "table"),
        ("table", "image"),
        ("image", "table"),
    }
    assert edges[("table", "text")]["candidate_ids"] == [
        "positive_text",
        "negative_text",
    ]
    assert edges[("table", "image")]["candidate_ids"] == [
        "positive_image",
        "negative_image",
    ]
    assert edges[("text", "table")]["query_id"] == "positive_text"
    assert edges[("text", "table")]["candidate_ids"] == ["positive", "negative"]
    assert edges[("image", "table")]["query_id"] == "positive_image"
    assert edges[("image", "table")]["candidate_ids"] == ["positive", "negative"]


def test_stage1_constructor_resolves_raw_data_lake_source_references(tmp_path):
    query = _table("q", ["Entity"], [["A"]])
    positive = _table("positive", ["Value"], [["1"]])
    source = _table("unused", ["Entity", "Other"], [["B", "2"]])
    source.pop("table_id")
    source.pop("object_id")
    source["source_table_id"] = "source_raw"
    raw_target = {
        "table_id": "raw",
        "object_id": "raw",
        "object_type": "table",
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

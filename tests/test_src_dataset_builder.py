from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_dataset.extraction import build_extractions
from mmdd_dataset.joinability import BuildConfig, build_joinability_dataset
from mmdd_dataset.pipeline import main as pipeline_main
from mmdd_dataset.tables import prepare_entitables, prepare_wdc
from mmdd_dataset.utils import write_jsonl
from mmdd_dataset.workload import generate_query_views
from build_image_attribute_dataset import build as build_image_attribute_dataset
from build_table_dataset import main as table_pipeline_main


def write_entitables(path: Path, table_count: int = 1, row_count: int = 5) -> None:
    payload = {}
    for table_index in range(table_count):
        payload[f"table-{table_index}"] = {
            "title": ["Entity", "Founded", "Category"],
            "pgTitle": f"Page {table_index}",
            "numCols": 3,
            "data": [
                [
                    f"[Entity_{table_index}_{row}|Entity {table_index} {row}]",
                    str(1900 + row),
                    f"Category {row % 2}",
                ]
                for row in range(row_count)
            ],
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def synthetic_materials(prepared):
    assets = []
    extractions = []
    entity_by_title = {entity["wiki_title"]: entity for entity in prepared.entities}
    table = prepared.source_tables[0]
    for row in table["rows"]:
        entity = entity_by_title[row["cells"][0]["wiki_title"]]
        asset_id = f"asset-{row['row_id']}"
        assets.append(
            {
                "asset_id": asset_id,
                "entity_id": entity["entity_id"],
                "asset_type": "text",
                "content": "synthetic evidence",
            }
        )
        extractions.append(
            {
                "source_table_id": table["source_table_id"],
                "source_row_id": row["row_id"],
                "entity_id": entity["entity_id"],
                "asset_id": asset_id,
                "asset_type": "text",
                "attribute_name": "Founded",
                "value": row["cells"][1]["text"] if row["row_id"] < 3 else "wrong",
                "evidence": "synthetic evidence",
            }
        )
    return assets, extractions


def test_entitables_adapter_and_joinability_core(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_entitables(input_dir / "tables.json")
    prepared = prepare_entitables(input_dir)
    assets, extractions = synthetic_materials(prepared)
    table = prepared.source_tables[0]

    assert table["metadata"]["candidate_entity_columns"] == [0]
    result = build_joinability_dataset(
        prepared.source_tables,
        assets,
        extractions,
        {table["source_table_id"]: "train"},
        BuildConfig(
            query_rows=5,
            min_target_rows=5,
            min_recovered_ratio=0.6,
            min_recovered_rows=3,
        ),
    )

    assert len(result["query_tables"]) == 1
    assert len(result["data_lake_tables"]) == 1
    assert len(result["qrels"]) == 1
    query, target = result["query_tables"][0], result["data_lake_tables"][0]
    assert "Founded" not in [column["column_name"] for column in query["columns"]]
    assert target["columns"][0]["column_name"] == "Founded"
    assert len(result["evidence_recoveries"]) == 3
    assert result["qrels"][0]["target_table_id"] == target["table_id"]


def test_table_workload_projects_reproducible_query_views(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    write_entitables(input_dir / "tables.json")
    prepared = prepare_entitables(input_dir)

    views = generate_query_views(prepared.source_tables[0], max_views=4, seed=13)
    assert len(views) == 4
    assert all(0 in view["selected_column_indices"] for view in views)
    assert all(
        [cell["column_index"] for cell in view["rows"][0]["cells"]]
        == list(range(len(view["selected_column_indices"])))
        for view in views
    )

    assert table_pipeline_main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--max-query-views-per-table",
            "4",
        ]
    ) == 0
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text())
    assert manifest["artifacts"]["query_views"]["records"] == 4


def test_extraction_masks_the_requested_attribute(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_entitables(input_dir / "tables.json")
    prepared = prepare_entitables(input_dir)
    table = prepared.source_tables[0]
    first_entity = prepared.entities[0]
    assets = [
        {
            "asset_id": "asset-1",
            "entity_id": first_entity["entity_id"],
            "asset_type": "text",
            "content": "Evidence",
        }
    ]

    calls = []

    class Extractor:
        def extract(self, **kwargs):
            calls.append(kwargs)
            return {"value": "", "evidence": ""}

    build_extractions(
        [table],
        prepared.entities,
        assets,
        Extractor(),
        min_column_non_empty_ratio=0.5,
    )

    assert calls
    for call in calls:
        context_names = {item["name"] for item in call["context"]}
        assert call["attribute"] not in context_names


def test_recovery_threshold_is_defined_on_query_size(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_entitables(input_dir / "tables.json", row_count=20)
    prepared = prepare_entitables(input_dir)
    assets, extractions = synthetic_materials(prepared)
    table = prepared.source_tables[0]

    result = build_joinability_dataset(
        prepared.source_tables,
        assets,
        extractions,
        {table["source_table_id"]: "train"},
        BuildConfig(query_rows=5, min_recovered_ratio=0.6, min_recovered_rows=3),
    )

    assert len(result["query_tables"]) == 1
    hidden = result["query_tables"][0]["hidden_attributes"][0]
    assert hidden["recovered_rows"] == 3
    assert hidden["recovered_value_ratio"] == 0.6


def test_wdc_adapter_uses_the_same_source_table_schema(tmp_path: Path) -> None:
    input_dir = tmp_path / "wdc"
    path = input_dir / "Thing" / "Thing_example.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in range(3):
            handle.write(
                json.dumps(
                    {
                        "row_id": 10 + row,
                        "name": f"Entity {row}",
                        "geo": {"lat": row},
                        "image": f"/image-{row}.jpg",
                        "page_url": f"https://example.test/{row}",
                    }
                )
                + "\n"
            )

    prepared = prepare_wdc(input_dir, min_rows=3, min_cols=2)
    table = prepared.source_tables[0]

    assert table["metadata"]["candidate_entity_columns"] == [0]
    assert [column["column_name"] for column in table["columns"]] == [
        "name",
        "geo",
        "page_url",
    ]
    assert table["rows"][0]["cells"][1]["text"] == '{"lat":0}'
    assert prepared.entities[0]["image_urls"] == ["https://example.test/image-0.jpg"]


def test_offline_cli_pipeline_is_self_contained(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    write_entitables(input_dir / "tables.json")
    prepared = prepare_entitables(input_dir)
    assets, extractions = synthetic_materials(prepared)
    assets_path = tmp_path / "assets.jsonl"
    extractions_path = tmp_path / "extractions.jsonl"
    write_jsonl(assets_path, assets)
    write_jsonl(extractions_path, extractions)

    exit_code = pipeline_main(
        [
            "--source",
            "entitables",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--assets-jsonl",
            str(assets_path),
            "--extractions-jsonl",
            str(extractions_path),
        ]
    )

    manifest = json.loads((output_dir / "dataset_manifest.json").read_text())
    assert exit_code == 0
    assert manifest["artifacts"]["query_tables"]["records"] == 1
    assert manifest["artifacts"]["qrels"]["records"] == 1


def test_image_attribute_builder_copies_selected_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    input_dir = tmp_path / "joinability"
    output_dir = tmp_path / "image_attributes"
    raw_dir.mkdir()
    input_dir.mkdir()
    write_entitables(raw_dir / "tables.json")
    prepared = prepare_entitables(raw_dir)
    table = prepared.source_tables[0]
    entity = prepared.entities[0]
    image_path = tmp_path / "evidence.jpg"
    image_path.write_bytes(b"synthetic-image")
    asset = {
        "asset_id": "image-asset",
        "entity_id": entity["entity_id"],
        "asset_type": "image",
        "local_path": str(image_path),
    }
    extraction = {
        "source_table_id": table["source_table_id"],
        "source_row_id": 0,
        "entity_id": entity["entity_id"],
        "asset_id": asset["asset_id"],
        "attribute_name": "Founded",
        "value": "1900",
        "evidence": "synthetic evidence",
    }
    write_jsonl(input_dir / "source_tables.jsonl", prepared.source_tables)
    write_jsonl(input_dir / "bridge_assets.jsonl", [asset])
    write_jsonl(input_dir / "attribute_extractions.jsonl", [extraction])

    stats = build_image_attribute_dataset(input_dir, output_dir, None, seed=13)

    assert stats["samples"] == 1
    assert stats["positive"] == 1
    assert len(list((output_dir / "images").iterdir())) == 1

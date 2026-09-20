from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mmdd_dataset.workload as workload_module
from build_image_attribute_dataset import build as build_image_attribute_dataset
from build_table_dataset import main as table_pipeline_main
from mmdd_dataset.extraction import (
    OpenAICompatibleExtractor,
    auto_check_recoveries,
    build_extractions,
)
from mmdd_dataset.joinability import (
    BuildConfig,
    _exact_redundancy_groups,
    _project,
    build_joinability_dataset,
)
from mmdd_dataset.pipeline import main as pipeline_main
from mmdd_dataset.tables import prepare_entitables, prepare_wdc
from mmdd_dataset.utils import sanitize_cell_text, write_jsonl
from mmdd_dataset.workload import generate_query_views


def write_entitables(path: Path, table_count: int = 1, row_count: int = 5) -> None:
    payload = {}
    for table_index in range(table_count):
        payload[f"table-{table_index}"] = {
            "title": ["Entity", "Founded", "Category", "Headquarters"],
            "pgTitle": f"Page {table_index}",
            "numCols": 4,
            "data": [
                [
                    f"[Entity_{table_index}_{row}|Entity {table_index} {row}]",
                    str(1900 + row),
                    f"Category {row % 2}",
                    f"City {row % 2}",
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


def write_custom_table(path: Path, title: list[str], rows: list[list[str]]) -> None:
    payload = {
        "table-0": {
            "title": title,
            "pgTitle": "Page 0",
            "numCols": len(title),
            "data": rows,
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def column_extractions(prepared, matches: dict[str, int]):
    """Assets per row plus extraction records matching `matches`.

    `matches` maps an attribute (column) name to the number of leading rows
    whose cell value the extraction reproduces.
    """
    table = prepared.source_tables[0]
    entity_by_title = {entity["wiki_title"]: entity for entity in prepared.entities}
    columns = {
        column["column_name"]: column["column_index"] for column in table["columns"]
    }
    assets = []
    extractions = []
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
        for attribute_name, match_rows in matches.items():
            column_index = columns[attribute_name]
            extractions.append(
                {
                    "source_table_id": table["source_table_id"],
                    "source_row_id": row["row_id"],
                    "entity_id": entity["entity_id"],
                    "asset_id": asset_id,
                    "asset_type": "text",
                    "attribute_name": attribute_name,
                    "value": (
                        row["cells"][column_index]["text"]
                        if row["row_id"] < match_rows
                        else "wrong"
                    ),
                    "evidence": "synthetic evidence",
                }
            )
    return assets, extractions


def build_with(
    prepared, assets, extractions, **config_overrides
):
    table = prepared.source_tables[0]
    return build_joinability_dataset(
        prepared.source_tables,
        assets,
        extractions,
        {table["source_table_id"]: "train"},
        BuildConfig(
            query_rows=5,
            min_target_rows=5,
            min_recovered_ratio=0.6,
            min_recovered_rows=3,
            **config_overrides,
        ),
    )


def test_entitables_adapter_and_joinability_core(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_entitables(input_dir / "tables.json")
    prepared = prepare_entitables(input_dir)
    assets, extractions = synthetic_materials(prepared)
    table = prepared.source_tables[0]

    assert table["metadata"]["candidate_entity_columns"] == [0]
    assert all(
        set(profile)
        == {
            "column_index",
            "non_empty_ratio",
            "wiki_link_ratio",
            "unique_ratio",
            "numeric_ratio",
        }
        for profile in table["metadata"]["column_profiles"]
    )
    assert all(
        set(column) == {"column_index", "column_name"}
        for column in table["columns"]
    )
    assert not {"page_title", "caption", "section_title"} & table.keys()
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
    assert query["split"] == "train"
    assert "split" not in target
    assert not {"page_title", "caption", "section_title"} & query.keys()
    assert not {"page_title", "caption", "section_title"} & target.keys()
    assert "Founded" not in [column["column_name"] for column in query["columns"]]
    assert target["columns"][0]["column_name"] == "Founded"
    assert len(result["evidence_recoveries"]) == 3
    assert {record["split"] for record in result["evidence_recoveries"]} == {"train"}
    assert result["qrels"][0]["target_table_id"] == target["table_id"]
    assert "data_lake_table_id" not in result["qrels"][0]
    assert "object_id" not in query
    assert "object_type" not in query
    assert "entity_url" in [column["column_name"] for column in query["columns"]]
    assert query["rows"][0]["cells"][-1]["synthetic"] is True
    assert query["rows"][0]["cells"][-1]["text"].startswith(
        "https://en.wikipedia.org/wiki/"
    )


def test_exact_redundancy_groups_are_row_aligned_and_length_delimited() -> None:
    assert _exact_redundancy_groups({0: ["a", "b"], 1: ["a", "b"]}) == [[0, 1]]
    assert _exact_redundancy_groups({0: ["a", "b"], 1: ["a", "c"]}) == []
    assert _exact_redundancy_groups({0: ["a", "b"], 1: ["b", "a"]}) == []
    assert _exact_redundancy_groups({0: ["", "a"], 1: ["a", ""]}) == []
    assert _exact_redundancy_groups({0: ["ab", "c"], 1: ["a", "bc"]}) == []
    # Serializer-defined equality: whitespace differences collapse to one group.
    assert (
        _exact_redundancy_groups(
            {0: ["a  b"], 1: ["a b"]}, value_serializer=sanitize_cell_text
        )
        == [[0, 1]]
    )
    # Distinct URLs stay distinct under the serializer.
    assert (
        _exact_redundancy_groups(
            {0: ["https://a.test/x", "same"], 1: ["https://b.test/x", "same"]},
            value_serializer=sanitize_cell_text,
        )
        == []
    )


def test_redundant_bridge_group_shares_one_query_and_fans_out_targets(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_custom_table(
        input_dir / "tables.json",
        ["Entity", "Bridge A", "Bridge B", "Context C", "Context D"],
        [
            [
                f"[Ent_{row}|Entity {row}]",
                f"Value {row}",
                f"Value {row}",
                f"C {row % 2}",
                f"D {row % 2}",
            ]
            for row in range(5)
        ],
    )
    prepared = prepare_entitables(input_dir)
    assets, extractions = column_extractions(
        prepared, {"Bridge A": 3, "Bridge B": 3}
    )

    result = build_with(prepared, assets, extractions)

    queries = result["query_tables"]
    targets = result["data_lake_tables"]
    qrels = result["qrels"]
    assert len(queries) == 1
    assert 1 <= len(targets) == len(qrels) <= 2
    query = queries[0]
    assert not {1, 2} & set(query["source_column_indices"])
    assert len(set(query["source_column_indices"]) - {0}) == 1
    assert query["target_table_ids"] == [target["table_id"] for target in targets]
    assert len(query["hidden_attributes"]) == len(targets)
    assert {hidden["column_name"] for hidden in query["hidden_attributes"]} <= {
        "Bridge A",
        "Bridge B",
    }
    assert len({qrel["chain_id"] for qrel in qrels}) == len(qrels)
    for qrel, target in zip(qrels, targets):
        assert target["join_col_name"] == qrel["join_attribute"]["column_name"]
        assert target["join_col_name"] in {"Bridge A", "Bridge B"}
    member_recoveries = [
        recovery
        for recovery in result["evidence_recoveries"]
    ]
    assert len(member_recoveries) == 3 * len(targets)
    for recovery in member_recoveries:
        assert recovery["recovered_attribute"]["column_name"] in {
            "Bridge A",
            "Bridge B"
        }
        assert recovery["recovered_attribute"]["value"].startswith("Value ")
        assert recovery["target_table_id"] in query["target_table_ids"]


def test_entity_alias_group_is_not_a_hidden_bridge(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_custom_table(
        input_dir / "tables.json",
        ["Entity", "Entity alias", "Bridge", "Context C", "Context D"],
        [
            [
                f"[Ent_{row}|Entity {row}]",
                f"Entity {row}",
                f"Value {row}",
                f"C {row % 2}",
                f"D {row % 2}",
            ]
            for row in range(5)
        ],
    )
    prepared = prepare_entitables(input_dir)
    assets, extractions = column_extractions(prepared, {"Bridge": 3})

    result = build_with(prepared, assets, extractions)

    assert len(result["query_tables"]) == 1
    query = result["query_tables"][0]
    # The alias column is neither a bridge nor context anywhere.
    assert not {1, 2} & set(query["source_column_indices"])
    assert [hidden["column_name"] for hidden in query["hidden_attributes"]] == [
        "Bridge"
    ]
    decision = result["table_queryability_decisions"][0]
    assert decision["reason"] == "queryable"
    assert [item["column_name"] for item in decision["qualified_columns"]] == [
        "Bridge"
    ]


def test_implicit_context_floor_rejects_narrow_tables(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_custom_table(
        input_dir / "tables.json",
        ["Entity", "Founded", "Category"],
        [
            [
                f"[Ent_{row}|Entity {row}]",
                str(1900 + row),
                f"Category {row % 2}",
            ]
            for row in range(5)
        ],
    )
    prepared = prepare_entitables(input_dir)
    assets, extractions = column_extractions(prepared, {"Founded": 3})

    result = build_with(prepared, assets, extractions)

    assert result["query_tables"] == []
    assert result["qrels"] == []
    decision = result["table_queryability_decisions"][0]
    assert decision["reason"] == "context_floor_unreachable"
    assert [item["column_name"] for item in decision["qualified_columns"]] == [
        "Founded"
    ]


def test_implicit_context_floor_demotes_weakest_bridge(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_custom_table(
        input_dir / "tables.json",
        ["Entity", "Strong Bridge", "Weak Bridge", "Only Context"],
        [
            [
                f"[Ent_{row}|Entity {row}]",
                f"Value {row}",
                f"Other {row}",
                f"C {row % 2}",
            ]
            for row in range(5)
        ],
    )
    prepared = prepare_entitables(input_dir)
    assets, extractions = column_extractions(
        prepared, {"Strong Bridge": 4, "Weak Bridge": 3}
    )

    result = build_with(prepared, assets, extractions)

    assert len(result["query_tables"]) == 1
    query = result["query_tables"][0]
    assert 1 not in set(query["source_column_indices"])
    context_columns = set(query["source_column_indices"]) - {0}
    assert context_columns <= {2, 3}
    target = result["data_lake_tables"][0]
    target_columns = {
        cell["source_column_index"] for cell in target["rows"][0]["cells"]
    }
    assert target_columns - {1} == {2, 3} - context_columns
    decision = result["table_queryability_decisions"][0]
    assert decision["reason"] == "queryable"
    assert [item["column_name"] for item in decision["qualified_columns"]] == [
        "Strong Bridge"
    ]


def test_projected_cell_text_is_truncated_and_keeps_urls() -> None:
    long_url = "https://example.test/" + "a" * 2000
    table = {
        "source_table_id": "st_x",
        "columns": [
            {"column_index": 0, "column_name": "A"},
            {"column_index": 1, "column_name": "B"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    {"column_index": 0, "text": "alpha"},
                    {"column_index": 1, "text": f"See {long_url} " + "x" * 2000},
                ],
            }
        ],
        "metadata": {"column_profiles": [], "candidate_entity_columns": []},
    }

    rows, _ = _project(table, [0, 1], {0})
    cell = rows[0]["cells"][1]
    assert len(cell["text"]) == 1024
    assert "https://example.test/" in cell["text"]
    assert sanitize_cell_text("  spaced  ") == "spaced"
    assert sanitize_cell_text(None) == ""


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
    splits = json.loads((output_dir / "splits.json").read_text())
    assert manifest["format"] == "mmdd_table_workload_research_v2"
    assert manifest["artifacts"]["query_views"]["records"] == 4
    assert splits["split_policy"] == "query_only"
    assert splits["data_lake_scope"] == "shared"
    assert splits["data_lake_source_table_count"] == 1
    assert splits["data_lake_artifact"] == "source_tables"
    assert sum(splits["query_view_counts"].values()) == 4


def test_table_workload_skips_random_projections_when_deterministic_views_fill_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_entitables(input_dir / "tables.json")
    prepared = prepare_entitables(input_dir)
    monkeypatch.setattr(
        workload_module,
        "combinations",
        lambda *_args, **_kwargs: pytest.fail("random projections were enumerated"),
    )

    views = generate_query_views(prepared.source_tables[0], max_views=1, seed=13)

    assert len(views) == 1
    assert views[0]["derivation_type"] != "random_projection"


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
        visible_names = {item["name"] for item in call["visible_cells"]}
        assert call["attribute"] not in visible_names


def test_src_auto_check_uses_only_materialized_query_columns(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_entitables(input_dir / "tables.json")
    prepared = prepare_entitables(input_dir)
    assets, extractions = synthetic_materials(prepared)
    table = prepared.source_tables[0]
    artifacts = build_joinability_dataset(
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
    local_calls: list[dict] = []
    luna_calls: list[dict] = []
    terra_calls: list[dict] = []

    class Extractor:
        def __init__(self, calls):
            self.calls = calls

        def extract(self, **kwargs):
            self.calls.append(kwargs)
            visible = {
                item["name"]: item["value"]
                for item in kwargs["visible_cells"]
            }
            row = int(visible["Entity"].rsplit(" ", 1)[1])
            return {"value": str(1900 + row), "evidence": ""}

    checked = auto_check_recoveries(
        artifacts,
        assets,
        Extractor(local_calls),
        luna_extractor=Extractor(luna_calls),
        terra_extractor=Extractor(terra_calls),
    )

    assert len(checked["evidence_recoveries"]) == 3
    assert local_calls
    assert len(luna_calls) == len(local_calls)
    assert terra_calls == []
    assert all(
        {cell["name"] for cell in call["visible_cells"]}
        == {"Entity", "Category", "entity_url"}
        for call in local_calls + luna_calls
    )
    assert all(
        call["visible_cells"][-1]["name"] == "entity_url"
        and call["visible_cells"][-1]["value"].startswith("https://en.wikipedia.org/wiki/")
        for call in local_calls + luna_calls
    )
    assert all(
        "Founded" not in {cell["name"] for cell in call["visible_cells"]}
        for call in local_calls + luna_calls
    )
    assert all(
        recovery["auto_check"]["decision_source"]
        == "local_luna_consensus"
        for recovery in checked["evidence_recoveries"]
    )


def test_src_auto_check_supports_local_only_and_terra_adjudication(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_entitables(input_dir / "tables.json")
    prepared = prepare_entitables(input_dir)
    assets, extractions = synthetic_materials(prepared)
    table = prepared.source_tables[0]
    artifacts = build_joinability_dataset(
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

    class Extractor:
        def __init__(self, mode: str):
            self.mode = mode
            self.calls = 0

        def extract(self, **kwargs):
            self.calls += 1
            if self.mode == "wrong":
                return {"value": "wrong", "evidence": ""}
            visible = {
                item["name"]: item["value"]
                for item in kwargs["visible_cells"]
            }
            row = int(visible["Entity"].rsplit(" ", 1)[1])
            return {"value": str(1900 + row), "evidence": ""}

    local_only = Extractor("correct")
    local_checked = auto_check_recoveries(
        artifacts,
        assets,
        local_only,
        review_mode="local",
    )
    assert len(local_checked["evidence_recoveries"]) == 3
    assert all(
        recovery["auto_check"]["decision_source"] == "primary_local"
        for recovery in local_checked["evidence_recoveries"]
    )

    local = Extractor("correct")
    luna = Extractor("wrong")
    terra = Extractor("correct")
    cascade_checked = auto_check_recoveries(
        artifacts,
        assets,
        local,
        luna_extractor=luna,
        terra_extractor=terra,
    )
    assert len(cascade_checked["evidence_recoveries"]) == 3
    assert local.calls == luna.calls == terra.calls
    assert all(
        recovery["auto_check"]["decision_source"] == "terra_adjudication"
        for recovery in cascade_checked["evidence_recoveries"]
    )


def test_src_model_message_omits_asset_metadata(monkeypatch) -> None:
    sent: dict = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"value":"1900","evidence":""}'}}
                ]
            }

    extractor = OpenAICompatibleExtractor("http://model.test/v1", "model")

    def post(_url, *, json, timeout):
        sent.update(json)
        return Response()

    monkeypatch.setattr(extractor.session, "post", post)
    extractor.extract(
        attribute="Founded",
        visible_cells=[{"name": "Entity", "value": "Alpha"}],
        asset={
            "asset_type": "text",
            "asset_id": "SECRET ASSET ID",
            "title": "SECRET TITLE",
            "source": "SECRET SOURCE",
            "url": "https://secret.example",
            "content": "Alpha was founded in 1900.",
        },
    )

    rendered = json.dumps(sent["messages"], ensure_ascii=False)
    assert "Alpha was founded in 1900." in rendered
    assert "evidence may be unrelated to the entity" in rendered
    assert "merely assuming the entity-evidence relationship" in rendered
    assert "SECRET ASSET ID" not in rendered
    assert "SECRET TITLE" not in rendered
    assert "SECRET SOURCE" not in rendered
    assert "secret.example" not in rendered


def test_src_image_message_uses_only_table_and_image(
    tmp_path: Path, monkeypatch
) -> None:
    sent: dict = {}
    image_path = tmp_path / "evidence.png"
    image_path.write_bytes(b"synthetic-image-bytes")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"value":"1900","evidence":""}'}}
                ]
            }

    extractor = OpenAICompatibleExtractor("http://model.test/v1", "model")

    def post(_url, *, json, timeout):
        sent.update(json)
        return Response()

    monkeypatch.setattr(extractor.session, "post", post)
    extractor.extract(
        attribute="Founded",
        visible_cells=[{"name": "Entity", "value": "Alpha"}],
        asset={
            "asset_type": "image",
            "asset_id": "SECRET IMAGE ID",
            "title": "SECRET IMAGE TITLE",
            "source": "SECRET IMAGE SOURCE",
            "content": "SECRET IMAGE CAPTION",
            "local_path": str(image_path),
        },
    )

    rendered = json.dumps(sent["messages"], ensure_ascii=False)
    assert "data:image/png;base64," in rendered
    assert "SECRET IMAGE ID" not in rendered
    assert "SECRET IMAGE TITLE" not in rendered
    assert "SECRET IMAGE SOURCE" not in rendered
    assert "SECRET IMAGE CAPTION" not in rendered
    assert str(image_path) not in rendered


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


def test_a_dropped_connection_is_retried_instead_of_ending_the_run() -> None:
    """One transient drop must not throw away a long extraction pass.

    A tunnel or a loaded engine closes a connection every so often.  The pass
    that hit this was six hours in and wrote nothing, because the failure
    surfaced as an exception out of ``extract`` rather than as a retry.
    """
    import requests

    attempts = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"value":"1900","evidence":""}'}}]}

    extractor = OpenAICompatibleExtractor("http://model.test/v1", "model",
                                          retry_sleep_seconds=0.0)

    def post(_url, *, json, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.exceptions.ConnectionError("Remote end closed connection")
        return Response()

    extractor.session.post = post
    result = extractor.extract(
        attribute="Founded",
        visible_cells=[{"name": "Entity", "value": "Alpha"}],
        asset={"asset_type": "text", "asset_id": "ev:1", "content": "Evidence."},
    )

    assert result["value"] == "1900"
    assert len(attempts) == 3, "the drop was not retried"


def test_a_rejected_request_is_not_retried() -> None:
    """A 4xx is an answer, not a hiccup: retrying it only burns the budget."""
    import requests

    attempts = []

    class Response:
        status_code = 400

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

        def json(self):
            return {}

    extractor = OpenAICompatibleExtractor("http://model.test/v1", "model",
                                          retry_sleep_seconds=0.0)

    def post(_url, *, json, timeout):
        attempts.append(1)
        return Response()

    extractor.session.post = post
    with pytest.raises(requests.exceptions.HTTPError):
        extractor.extract(
            attribute="Founded",
            visible_cells=[{"name": "Entity", "value": "Alpha"}],
            asset={"asset_type": "text", "asset_id": "ev:1", "content": "Evidence."},
        )
    assert len(attempts) == 1, "a rejected request was retried"


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


def test_wdc_queries_omit_the_synthetic_entity_url_column(tmp_path: Path) -> None:
    """The synthetic ``entity_url`` is a Wikipedia URL built from ``wiki_title``.

    WDC mints a ``wdc_<hash>`` title rather than a real article title, so the
    column would hold a fabricated ``en.wikipedia.org`` link that reads as
    evidence and is not.  The WDC entry points therefore turn it off.
    """
    input_dir = tmp_path / "wdc"
    path = input_dir / "Organization" / "Organization_example.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in range(5):
            handle.write(
                json.dumps(
                    {
                        "row_id": row,
                        "name": f"Entity {row}",
                        "founded": str(1900 + row),
                        "location": f"City {row}",
                        "industry": f"Industry {row}",
                        "page_url": f"https://example.test/{row}",
                    }
                )
                + "\n"
            )

    prepared = prepare_wdc(input_dir, min_rows=5, min_cols=2)
    assets, extractions = synthetic_materials(prepared)

    with_column = build_with(prepared, assets, extractions)
    without_column = build_with(
        prepared, assets, extractions, synthetic_entity_url=False
    )

    assert len(with_column["query_tables"]) == 1
    assert len(without_column["query_tables"]) == 1
    fabricated = with_column["query_tables"][0]
    assert fabricated["columns"][-1]["column_name"] == "entity_url"
    assert fabricated["rows"][0]["cells"][-1]["text"].startswith(
        "https://en.wikipedia.org/wiki/wdc_"
    )

    query = without_column["query_tables"][0]
    assert "entity_url" not in [column["column_name"] for column in query["columns"]]
    assert '"column_name": "entity_url"' not in json.dumps(query)


def test_abebooks_adapter_fills_the_entity_slot_without_wikipedia(tmp_path: Path) -> None:
    """The shared gates read ``wiki_title``; AbeBooks has no Wikipedia.

    That field is only the row -> entity pointer -- the entity-column gate counts
    non-empty ones and the asset linkage looks entities up by it -- so an opaque
    ``abe_<hash>`` satisfies it exactly as WDC's ``wdc_<hash>`` does.  The row id
    becomes an integer because the shared builder keys tasks and join records
    that way, and the lake's ``bk_0001`` string stays recoverable for audit.
    """
    from mmdd_dataset.abebooks_adapter import prepare_abebooks

    lake = tmp_path / "lake"
    lake.mkdir()
    write_jsonl(lake / "source_tables.jsonl", [_synthetic_lake_table()])

    prepared = prepare_abebooks(lake, min_rows=2, min_cols=2)

    assert prepared.skipped == {}
    table = prepared.source_tables[0]
    assert sorted(table) == ["columns", "metadata", "num_cols", "num_rows",
                             "provenance_builder", "rows", "source_file",
                             "source_table_id"]
    entity_column = table["metadata"]["candidate_entity_columns"][0]
    assert table["columns"][entity_column]["column_name"] == "title"
    first = table["rows"][0]
    assert isinstance(first["row_id"], int), "the shared builder keys rows by int"
    key = first["cells"][entity_column]["wiki_title"]
    assert key.startswith("abe_") and first["cells"][entity_column]["has_wiki_link"]
    assert table["metadata"]["column_profiles"][entity_column]["wiki_link_ratio"] == 1.0
    # The two things ``choose_entity_column`` reads: a non-empty wiki_title on
    # every row's entity cell, and a wiki-link ratio the gate accepts.  (Verified
    # separately against the real 130-table lake: it accepts all of them.)
    assert all(row["cells"][entity_column]["wiki_title"] for row in table["rows"])

    entity = prepared.entities[0]
    assert entity["wiki_title"] == key
    assert entity["source_row_id"] == 1
    assert entity["lake_row_id"] == "bk_0001", "the lake id stays for audit"

    # Declares itself as a corpus that mints its entity keys, which is what tells
    # the shared builder not to append a synthetic ``entity_url`` column: the URL
    # would be https://en.wikipedia.org/wiki/abe_<hash>, a fabricated page.  WDC
    # opts out the same way; EntiTables, which has real pages, does not.
    assert table["provenance_builder"] == "abebooks_mm_joinability_dataset"


def test_abebooks_adapter_links_assets_to_their_rows_entity(tmp_path: Path) -> None:
    """The lake keys assets by row; the shared builder reaches them by entity."""
    from mmdd_dataset.abebooks_adapter import adapt_assets, prepare_abebooks

    lake = tmp_path / "lake"
    (lake / "images").mkdir(parents=True)
    image = lake / "images" / "cover.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    table = _synthetic_lake_table()
    write_jsonl(lake / "source_tables.jsonl", [table])
    write_jsonl(lake / "bridge_assets.jsonl", [
        {"asset_id": "ev:text", "asset_type": "text", "source": "abebooks_synopsis",
         "source_column": "vendor_description", "row_id": "bk_0001",
         "content": "A text excerpt.", "url": "None", "local_path": "None"},
        {"asset_id": "ev:image", "asset_type": "image", "source": "abebooks_catalogue_cover",
         "source_column": None, "row_id": "bk_0002", "content": "None", "url": "None",
         "local_path": str(image.relative_to(tmp_path))},
    ])

    prepared = prepare_abebooks(lake, min_rows=2, min_cols=2)
    assets = adapt_assets(lake, root=tmp_path)

    by_id = {asset["asset_id"]: asset for asset in assets}
    assert by_id["ev:text"]["entity_id"] == prepared.entities[0]["entity_id"]
    assert by_id["ev:image"]["entity_id"] == prepared.entities[1]["entity_id"]
    assert by_id["ev:text"]["source_table_id"] == table["source_table_id"]
    assert by_id["ev:image"]["source_row_id"] == 2
    assert by_id["ev:image"]["content"] == "None", "a text field is left as the lake wrote it"
    # image paths are opened by the client, so they cannot stay relative
    assert Path(by_id["ev:image"]["local_path"]).is_absolute()
    assert Path(by_id["ev:image"]["local_path"]).exists()


def _synthetic_lake_table() -> dict:
    """A two-row lake table in the shape ``build_abebooks_lake.py`` writes."""

    def cell(index: int, name: str, text: str) -> dict:
        return {"column_index": index, "column_name": name, "raw": text, "text": text,
                "wiki_title": None, "has_wiki_link": False}

    columns = ["title", "authors", "publisher"]
    rows = [
        {"row_id": f"bk_{n:04d}",
         "cells": [cell(0, "title", f"Book {n}"), cell(1, "authors", f"Author {n}"),
                   cell(2, "publisher", "Acme")]}
        for n in (1, 2)
    ]
    return {
        "source_table_id": "st_book_001",
        "source_name": "book",
        "num_rows": len(rows),
        "num_cols": len(columns),
        "columns": [{"column_index": i, "column_name": name} for i, name in enumerate(columns)],
        "rows": rows,
        "metadata": {
            "candidate_entity_columns": [0],
            "column_profiles": [
                {"column_index": i, "non_empty_ratio": 1.0, "wiki_link_ratio": 0.0,
                 "unique_ratio": 1.0, "numeric_ratio": 0.0}
                for i in range(len(columns))
            ],
        },
    }


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
    splits = json.loads((output_dir / "splits.json").read_text())
    targets = list(
        json.loads(line)
        for line in (output_dir / "data_lake_tables.jsonl").read_text().splitlines()
    )
    assert exit_code == 0
    assert manifest["format"] == "mmdd_joinability_research_v2"
    assert manifest["split_schema_version"] == "query-only-shared-data-lake-v1"
    assert manifest["artifacts"]["query_tables"]["records"] == 1
    assert manifest["artifacts"]["qrels"]["records"] == 1
    assert all("split" not in target for target in targets)
    assert splits["split_policy"] == "query_only"
    assert splits["data_lake_scope"] == "shared"
    assert splits["data_lake_table_count"] == len(targets)
    assert splits["data_lake_artifact"] == "data_lake_tables"
    assert sum(splits["query_table_counts"].values()) == 1


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

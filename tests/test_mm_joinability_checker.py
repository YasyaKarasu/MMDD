import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mm_joinability_dataset_checker import (
    build_review_rows,
    create_checker_app,
    sampled_query_ids,
)
from stage1_io import write_jsonl


def table_record(table_id: str, role: str, value: str) -> dict:
    return {
        "table_id": table_id,
        "role": role,
        "split": "test",
        "source_table_id": f"source_{table_id}",
        "page_title": value,
        "caption": "synthetic checker fixture",
        "columns": [{"column_index": 0, "column_name": "Name"}],
        "rows": [
            {
                "row_id": 0,
                "source_row_id": 0,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "Name",
                        "text": value,
                    }
                ],
            }
        ],
    }


def write_checker_dataset(
    root: Path,
    *,
    implicit_queries: int = 100,
    ambiguous: bool = False,
) -> None:
    for artifact in (
        "query_tables",
        "data_lake_tables",
        "bridge_assets",
        "evidence_recoveries",
    ):
        (root / artifact).mkdir(parents=True)
    query_tables = []
    target_tables = []
    qrels = []
    for index in range(implicit_queries):
        query_id = f"query_{index:03d}"
        target_id = f"target_{index:03d}"
        query_tables.append(table_record(query_id, "query", f"Query {index}"))
        target_tables.append(table_record(target_id, "target_data_lake_table", f"Target {index}"))
        qrels.append(
            {
                "query_table_id": query_id,
                "target_table_id": target_id,
                "data_lake_table_id": target_id,
                "rel": 3,
                "split": "test",
                "source_table_id": f"source_{index:03d}",
                "join_attribute": {
                    "source_column_index": 1,
                    "column_name": "Country",
                },
                "reason": "model_recoverable_join_column",
            }
        )
    if ambiguous:
        target_tables.append(
            table_record("target_duplicate", "target_data_lake_table", "Duplicate")
        )
        qrels.append(
            {
                **qrels[0],
                "target_table_id": "target_duplicate",
                "data_lake_table_id": "target_duplicate",
                "join_attribute": {
                    "source_column_index": 2,
                    "column_name": "Year",
                },
            }
        )

    artifacts = {
        "query_tables": query_tables,
        "data_lake_tables": target_tables,
        "bridge_assets": [],
        "evidence_recoveries": [],
    }
    manifest_artifacts = {}
    for artifact, records in artifacts.items():
        relative = f"{artifact}/part-00000.jsonl"
        write_jsonl(root / relative, records)
        manifest_artifacts[artifact] = {
            "shards": [{"path": relative, "records": len(records)}]
        }
    write_jsonl(root / "qrels.jsonl", qrels)
    (root / "stats.json").write_text(
        json.dumps({"implicit_join_query_tables": implicit_queries}),
        encoding="utf-8",
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "format": "sharded_jsonl",
                "artifacts": manifest_artifacts,
                "single_files": {"qrels": "qrels.jsonl", "stats": "stats.json"},
            }
        ),
        encoding="utf-8",
    )


def test_checker_samples_one_percent_and_persists_final_quality_rate(
    tmp_path: Path,
) -> None:
    write_checker_dataset(tmp_path)
    review_db = tmp_path / "reviews.sqlite3"
    app = create_checker_app(
        tmp_path,
        dataset_name="Synthetic",
        sample_rate=0.01,
        seed=13,
        review_db=review_db,
        max_rows=2,
        max_paths=2,
        max_asset_chars=200,
    )
    store = app.config["QUALITY_CHECKER_STORE"]
    assert store.summary() == {
        "population": 100,
        "sampled": 1,
        "reviewed": 0,
        "qualified": 0,
        "unqualified": 0,
        "pending": 1,
        "complete": False,
        "current_rate": "—",
        "final_rate": "—",
    }

    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "唯一 attribute/target".encode() in response.data
    assert "Query Row → Evidence → Attribute → Target Row".encode() in response.data
    assert b'data-query-row-id="0"' in response.data
    page_html = response.data.decode("utf-8")
    assert 0 < page_html.index("完整 Query 表") < page_html.index("完整 Target 表")
    assert page_html.index("完整 Target 表") < page_html.index(
        "Query Row → Evidence → Attribute → Target Row"
    )
    query_id = store.query_id_at(1)
    response = client.post(
        "/review",
        data={
            "query_id": query_id,
            "page": "1",
            "rating": "qualified",
            "note": "looks good",
        },
    )
    assert response.status_code == 302
    assert store.summary()["final_rate"] == "100.00%"
    exported = client.get("/export").data.decode("utf-8")
    assert json.loads(exported)["rating"] == "qualified"

    reopened = create_checker_app(
        tmp_path,
        dataset_name="Synthetic",
        sample_rate=0.01,
        seed=13,
        review_db=review_db,
        max_rows=2,
        max_paths=2,
        max_asset_chars=200,
    )
    assert reopened.config["QUALITY_CHECKER_STORE"].summary()["reviewed"] == 1


def test_checker_rejects_ambiguous_implicit_query(tmp_path: Path) -> None:
    write_checker_dataset(tmp_path, implicit_queries=2, ambiguous=True)

    with pytest.raises(ValueError, match="one attribute/target"):
        create_checker_app(
            tmp_path,
            dataset_name="Ambiguous",
            review_db=tmp_path / "reviews.sqlite3",
        )


def test_seeded_sample_is_exact_and_order_independent() -> None:
    query_ids = [f"q{index}" for index in range(201)]
    first = sampled_query_ids(query_ids, 0.01, 7)
    second = sampled_query_ids(list(reversed(query_ids)), 0.01, 7)

    assert first == second
    assert len(first) == 3


def test_review_rows_align_each_query_row_with_its_evidence_and_attribute() -> None:
    pair = {
        "query_table": {
            "columns": ["Entity", "Club"],
            "rows": [
                {"row_id": 0, "cells": ["Alpha", "A FC"]},
                {"row_id": 1, "cells": ["Beta", "B FC"]},
            ],
        },
        "target_table": {
            "columns": ["Country"],
            "rows": [
                {"row_id": 10, "cells": ["Argentina"]},
                {"row_id": 11, "cells": ["France"]},
            ],
        },
        "paths": [
            {
                "path_id": "alpha_text",
                "query_row_id": "0",
                "target_row_ids": [10],
                "recovered_attribute": {"column_name": "Country", "value": "Argentina"},
            },
            {
                "path_id": "alpha_image",
                "query_row_id": "0",
                "target_row_ids": [10],
                "recovered_attribute": {"column_name": "Country", "value": "Argentina"},
            },
            {
                "path_id": "beta_text",
                "query_row_id": "1",
                "target_row_ids": [11],
                "recovered_attribute": {"column_name": "Country", "value": "France"},
            },
        ],
    }

    rows = build_review_rows(pair)

    assert [row["query_row_id"] for row in rows] == ["0", "0", "1"]
    assert [row["path"]["path_id"] for row in rows] == [
        "alpha_text",
        "alpha_image",
        "beta_text",
    ]
    assert [row["path"]["recovered_attribute"]["value"] for row in rows] == [
        "Argentina",
        "Argentina",
        "France",
    ]
    assert rows[2]["query_cells"] == [
        {"column": "Entity", "value": "Beta"},
        {"column": "Club", "value": "B FC"},
    ]
    assert rows[2]["target_rows"] == [
        {
            "row_id": "11",
            "cells": [{"column": "Country", "value": "France"}],
        }
    ]

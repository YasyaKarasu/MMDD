import gzip
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_wdc_mm_joinability_dataset import extract_image_urls, read_wdc_table
from stage1_io import get_cell, get_cell_text


def write_gzip_rows(tmp_path: Path, rows: list[Any], name: str = "Thing_host.json.gz") -> Path:
    path = tmp_path / "Thing" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row)
            else:
                handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return path


def test_extract_image_urls_recurses_resolves_and_deduplicates():
    value = {
        "contentUrl": "/a.jpg",
        "nested": [
            "https://cdn.test/b.png",
            {"url": "/a.jpg", "ignored": "data:image/png;base64,abc"},
        ],
    }

    assert extract_image_urls(value, "https://example.test/path/page") == [
        "https://example.test/a.jpg",
        "https://cdn.test/b.png",
    ]


def test_extract_image_urls_skips_malformed_urls_without_losing_valid_urls():
    assert extract_image_urls(
        ["http://[", "https://cdn.test/valid.jpg"],
        "https://example.test/page",
    ) == ["https://cdn.test/valid.jpg"]


def test_read_wdc_table_removes_image_and_preserves_nested_values(tmp_path):
    path = write_gzip_rows(
        tmp_path,
        [
            {
                "row_id": 7,
                "name": "A",
                "geo": {"lat": "1"},
                "image": ["/a.jpg"],
                "page_url": "https://x.test/p",
            },
            {
                "row_id": 8,
                "name": "B",
                "geo": {"lat": "2"},
                "image": "https://cdn.test/b.jpg",
                "page_url": "https://x.test/q",
            },
        ],
    )

    result = read_wdc_table(path, tmp_path, min_rows=2, min_cols=2)

    assert result.source_table is not None
    assert [column["column_name"] for column in result.source_table["columns"]] == [
        "name",
        "geo",
        "page_url",
    ]
    assert get_cell_text(result.source_table["rows"][0], 1) == '{"lat":"1"}'
    assert result.source_table["metadata"]["candidate_entity_columns"] == [0]
    assert len(result.source_table["metadata"]["column_profiles"]) == 3

    first_entity = result.entities[0]
    first_cell = get_cell(result.source_table["rows"][0], 0)
    assert first_cell["wiki_title"] == first_entity["wiki_title"]
    assert first_entity.keys() == {
        "entity_id",
        "wiki_title",
        "display_texts",
        "context_terms",
        "appears_in",
        "page_url",
        "image_urls",
    }
    assert result.image_urls_by_entity[first_entity["entity_id"]] == ["https://x.test/a.jpg"]
    assert first_entity["image_urls"] == ["https://x.test/a.jpg"]


def test_read_wdc_table_isolates_malformed_rows_and_honors_row_cap(tmp_path):
    path = write_gzip_rows(
        tmp_path,
        [
            "not-json",
            {"row_id": 3, "title": "First", "page_url": "https://x.test/shared"},
            ["not", "an", "object"],
            {
                "row_id": 4,
                "title": "Second",
                "late_column": "kept",
                "page_url": "https://x.test/shared",
            },
            {"row_id": 5, "title": "Not retained", "after_cap": "excluded"},
        ],
    )

    result = read_wdc_table(path, tmp_path, min_rows=2, min_cols=2, max_rows=2)

    assert result.source_table is not None
    assert result.malformed_rows == 2
    assert result.source_table["num_rows"] == 2
    assert [column["column_name"] for column in result.source_table["columns"]] == [
        "title",
        "page_url",
        "late_column",
    ]
    assert result.entities[0]["entity_id"] != result.entities[1]["entity_id"]


def test_read_wdc_table_isolates_invalid_utf8_lines(tmp_path):
    path = tmp_path / "Thing" / "Thing_host.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wb") as handle:
        handle.write(b"\xff\xfe\n")
        handle.write(b'{"name":"valid","page_url":"https://x.test/p"}\n')

    result = read_wdc_table(path, tmp_path, min_rows=1, min_cols=2)

    assert result.source_table is not None
    assert result.malformed_rows == 1


def test_read_wdc_table_rejects_tables_below_thresholds(tmp_path):
    path = write_gzip_rows(tmp_path, [{"name": "Only row", "image": "/only.jpg"}])

    too_few_rows = read_wdc_table(path, tmp_path, min_rows=2, min_cols=1)
    too_few_cols = read_wdc_table(path, tmp_path, min_rows=1, min_cols=2)

    assert too_few_rows.source_table is None
    assert too_few_rows.skip_reason is not None
    assert too_few_cols.source_table is None
    assert too_few_cols.skip_reason is not None

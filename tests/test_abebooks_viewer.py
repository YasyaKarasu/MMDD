"""Tests for the four-table HTML viewer.

The page is a build artifact with no logic of its own worth mocking, so these
tests pin the two things that would silently break it: the payload must survive
the trip into the ``<script>`` element intact, and it must carry every row of
every table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from build_abebooks_viewer import EXTRAS, PLACEHOLDER, TABLES, build, parser


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")


def payload_of(html: str) -> dict:
    """Pull the embedded JSON back out of the generated page."""
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>',
                      html, re.S)
    assert match, "the page lost its payload script element"
    return json.loads(match.group(1).replace("<\\/", "</"))


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    data = tmp_path / "abebooks_dataset"
    data.mkdir()
    write_jsonl(data / "book_edition.jsonl", [
        {"book_id": "isbn13:9780201616477", "title": "UNIX", "dimensions": None},
        {"book_id": "isbn13:9780201896848", "title": "TAOCP", "dimensions": "N/A"},
    ])
    write_jsonl(data / "seller.jsonl", [
        {"seller_id": "abebooks:3207104", "seller_name": "Book Lover's Warehouse"},
    ])
    write_jsonl(data / "book_listing.jsonl", [
        {"listing_id": "abebooks:1", "book_id": "isbn13:9780201616477",
         "seller_id": "abebooks:3207104", "price": 10.0},
    ])
    write_jsonl(data / "evidence_asset.jsonl", [
        {"evidence_id": "ev:aa", "book_id": "isbn13:9780201616477", "asset_type": "detail_page"},
        {"evidence_id": "ev:bb", "book_id": "isbn13:9780201616477", "asset_type": "search_page"},
    ])
    write_jsonl(data / "unresolved.jsonl", [
        {"isbn": "0201616475", "status": "degraded"},
    ])
    (data / "superseded_records.jsonl").write_text("", encoding="utf-8")
    (data / "stats.json").write_text(json.dumps({"counts": {"book_edition": 2}}), encoding="utf-8")
    (data / "dataset_manifest.json").write_text(json.dumps({
        "input": {"path": "output/abebooks_full.jsonl", "sha256": "a" * 64, "records": 289},
    }), encoding="utf-8")
    return data


def run(data_dir: Path, out_dir: Path) -> str:
    build(parser().parse_args(["--data-dir", str(data_dir), "--output-dir", str(out_dir)]))
    return (out_dir / "index.html").read_text(encoding="utf-8")


def test_every_table_and_row_reaches_the_page(dataset: Path, tmp_path: Path) -> None:
    payload = payload_of(run(dataset, tmp_path / "view"))
    by_name = {t["name"]: t for t in payload["tables"]}
    assert set(by_name) == set(TABLES) | {"unresolved"}
    assert [r["count"] for r in payload["tables"]] == [2, 1, 1, 2, 1]
    assert by_name["book_edition"]["columns"] == ["book_id", "title", "dimensions"]
    assert by_name["book_edition"]["rows"][0] == ["isbn13:9780201616477", "UNIX", None]


def test_empty_extra_file_is_skipped_rather_than_rendered_as_a_blank_tab(
    dataset: Path, tmp_path: Path
) -> None:
    payload = payload_of(run(dataset, tmp_path / "view"))
    assert "superseded_records" not in {t["name"] for t in payload["tables"]}
    assert "superseded_records" in EXTRAS


def test_stats_and_manifest_are_embedded(dataset: Path, tmp_path: Path) -> None:
    payload = payload_of(run(dataset, tmp_path / "view"))
    assert payload["stats"]["counts"]["book_edition"] == 2
    assert payload["input"]["records"] == 289
    assert payload["generated_at"].endswith("UTC")


def test_a_cell_containing_a_closing_script_tag_cannot_break_out(
    dataset: Path, tmp_path: Path
) -> None:
    """The one injection point: a cell whose text closes the script element.

    A ``<script type="application/json">`` body is never parsed as HTML, so
    markup in a cell is inert on its own -- the only way out is a literal
    ``</script``, which must be escaped.  The count of real closing tags is
    therefore the invariant: the payload element plus the app element.
    """
    poisoned = "</script><img src=x onerror=alert(1)>"
    write_jsonl(dataset / "book_edition.jsonl", [
        {"book_id": "isbn13:9780201616477", "title": poisoned, "dimensions": None},
    ])
    html = run(dataset, tmp_path / "view")
    blob = html.split('<script id="payload" type="application/json">')[1].split("</script>")[0]
    assert "</script" not in blob
    assert html.count("</script>") == 2
    assert payload_of(html)["tables"][0]["rows"][0][1] == poisoned


def test_a_column_added_partway_through_the_file_still_renders(
    dataset: Path, tmp_path: Path
) -> None:
    write_jsonl(dataset / "book_edition.jsonl", [
        {"book_id": "a", "title": "t"},
        {"book_id": "b", "title": "u", "extra": "late"},
    ])
    table = payload_of(run(dataset, tmp_path / "view"))["tables"][0]
    assert table["columns"] == ["book_id", "title", "extra"]
    assert table["rows"][0] == ["a", "t", None]  # missing key becomes NULL, not absent


def test_missing_book_edition_is_a_clear_failure(dataset: Path, tmp_path: Path) -> None:
    (dataset / "book_edition.jsonl").unlink()
    with pytest.raises(SystemExit, match="book_edition.jsonl"):
        run(dataset, tmp_path / "view")


def test_template_keeps_its_placeholder() -> None:
    template = Path(__file__).resolve().parents[1] / "scripts_old" / \
        "abebooks_viewer_template.html"
    assert PLACEHOLDER in template.read_text(encoding="utf-8")


def test_rebuilding_is_deterministic_apart_from_the_timestamp(
    dataset: Path, tmp_path: Path
) -> None:
    """Two builds of the same data must agree; only ``generated_at`` may differ."""
    first = payload_of(run(dataset, tmp_path / "view"))
    second = payload_of(run(dataset, tmp_path / "view"))
    first.pop("generated_at"), second.pop("generated_at")
    assert first == second

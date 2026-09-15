"""Tests for the two-table multimodal lake.

The scrape is a star schema -- every join key visible -- and the builder's whole
job is to break that without leaving the key behind somewhere else.  So these
tests are mostly leak tests: an ISBN in a column, an ISBN in a row id, or an
alias list for the very entity we hide would each make the join recoverable by
string matching instead of by evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_abebooks_lake import (BOOK_EDITION_COLUMNS, BOOK_LISTING_COLUMNS,
                                 SELLER_COLUMNS, _strip_boilerplate, build, chunk, parser,
                                 piece_counts)
from mmdd_dataset.workload import generate_query_views

ISBN = "9780201616477"
BOILERPLATE = ('"About the title" may belong to another edition of this title.')


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    data = tmp_path / "dataset"
    data.mkdir()
    write_jsonl(data / "book_edition.jsonl", [
        {"book_id": f"isbn13:{ISBN}", "isbn10": "0201616475", "isbn13": ISBN,
         "title": "UNIX System Administration Handbook", "authors": "Nemeth, Evi",
         "publisher": "Prentice Hall", "publication_year": 1995, "binding": "Paperback",
         "synopsis_text": f"Classic text. ISBN {ISBN} on the back.",
         "about_author_text": f"Evi Nemeth writes books. {ISBN}AB06262002 {BOILERPLATE}",
         "catalogue_image_url": f"https://pictures.abebooks.com/isbn/{ISBN}-us._SL300_.jpg",
         "source_url": "https://www.abebooks.com/UNIX-System-Admin-Nemeth/31595123263/bd"},
    ])
    write_jsonl(data / "book_listing.jsonl", [
        {"listing_id": "abebooks:31595123263", "book_id": f"isbn13:{ISBN}",
         "seller_id": "abebooks:3207104", "price": 12.5, "condition": "Used - Good",
         "seller_inventory_no": f"7719-{ISBN}",
         "vendor_image_url": f"https://pictures.abebooks.com/inventory/31595123263.jpg"},
    ])
    write_jsonl(data / "seller.jsonl", [
        {"seller_id": "abebooks:3207104", "seller_name": "Book Lover's Warehouse",
         "seller_name_variants": ["Book Lovers Warehouse", "BLW"],
         "seller_url": "https://www.abebooks.com/book-lovers-warehouse/3207104/sf",
         "city": "Nashville", "seller_rating": "5-star seller"},
    ])
    write_jsonl(data / "evidence_asset.jsonl", [
        {"evidence_id": "ev:img", "book_id": f"isbn13:{ISBN}", "asset_type": "seller_cover",
         "uri": "https://pictures.abebooks.com/inventory/31595123263.jpg"},
        {"evidence_id": "ev:txt", "book_id": f"isbn13:{ISBN}", "asset_type": "about_author",
         "record_isbn": ISBN, "extracted_text": f"Evi Nemeth writes books. {ISBN}AB06262002 {BOILERPLATE}"},
        {"evidence_id": "ev:policy", "book_id": f"isbn13:{ISBN}", "asset_type": "seller_policy",
         "listing_id": "abebooks:31595123263", "extracted_text": "Returns within 30 days."},
        {"evidence_id": "ev:page", "book_id": f"isbn13:{ISBN}", "asset_type": "detail_page",
         "uri": "https://www.abebooks.com/x/1/bd"},
    ])
    return data


def run(data: Path, tmp: Path, *extra: str) -> dict:
    args = parser().parse_args(["--data-dir", str(data), "--output-dir", str(tmp / "lake"),
                                "--image-manifest", str(tmp / "missing.jsonl"), *extra])
    return build(args)


def read_lake(out: Path) -> list[dict]:
    return [json.loads(line) for line in
            (out / "source_tables.jsonl").read_text(encoding="utf-8").splitlines()]


def test_no_column_carries_an_isbn_or_a_url(dataset: Path, tmp_path: Path) -> None:
    """The two leaks that survive simply dropping ``isbn13``.

    ``seller_inventory_no`` *is* the ISBN for 130/253 scraped rows, and the
    detail-page URL spells out a title/author slug -- which is the entity now.
    """
    columns = set(BOOK_EDITION_COLUMNS) | set(BOOK_LISTING_COLUMNS) | set(SELLER_COLUMNS)
    for banned in ("book_id", "isbn10", "isbn13", "seller_id", "seller_url",
                   "seller_inventory_no", "source_url", "listing_url",
                   "vendor_image_url", "catalogue_image_url", "seller_name_variants"):
        assert banned not in columns, f"{banned} would hand over the join key"


def test_no_cell_contains_the_isbn_or_a_url(dataset: Path, tmp_path: Path) -> None:
    run(dataset, tmp_path)
    blob = (tmp_path / "lake" / "source_tables.jsonl").read_text(encoding="utf-8")
    assert ISBN not in blob
    assert "0201616475" not in blob
    assert "http://" not in blob and "https://" not in blob


def test_row_ids_cannot_be_traced_back_to_the_isbn(dataset: Path, tmp_path: Path) -> None:
    """A key derived from the ISBN is recoverable by enumerating and hashing it."""
    run(dataset, tmp_path)
    tables = read_lake(tmp_path / "lake")
    row_ids = [row["row_id"] for table in tables for row in table["rows"]]
    assert row_ids == ["bk_0001", "sl_0001"]
    key_map = (tmp_path / "lake" / "key_map.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(key_map) == 2  # audit side only, never a model input


def test_the_boilerplate_and_the_isbn_it_carries_both_go(dataset: Path, tmp_path: Path) -> None:
    run(dataset, tmp_path)
    table = read_lake(tmp_path / "lake")[0]
    index = [c["column_name"] for c in table["columns"]].index("about_author_text")
    text = table["rows"][0]["cells"][index]["text"]
    assert text == "Evi Nemeth writes books."
    assert "About the title" not in text


def test_a_truncated_notice_is_still_removed(tmp_path: Path) -> None:
    """The upstream build caps cells at 1024 chars and cuts the notice in half.

    An anchored literal misses ``... And others. "About the title`` and leaves
    the ISBN sitting in the text, which is how the leak survives a fix that
    looks correct on untruncated rows.
    """
    truncated = f'Real bio text. {ISBN}AB06262002 "About the title'
    assert _strip_boilerplate(truncated) == "Real bio text."
    assert _strip_boilerplate('Real bio. "About the title" may belong to another edition.') == "Real bio."
    assert _strip_boilerplate("No notice here.") == "No notice here."
    # 91/175 notices are preceded by the token, and its letter code is not
    # always ``AB`` -- 4 rows carry a bare ``B``, which an ``AB``-only pattern
    # misses and so leaves the ISBN in the cell.
    assert _strip_boilerplate(f'{ISBN}AB04062001 "About the title" may belong.') is None
    assert _strip_boilerplate(f'{ISBN}B04062001 "About the title" may belong.') is None
    assert _strip_boilerplate(f'{ISBN}ABC04062001 "About the title" may belong.') is None
    # A bio that merely ends in a year must not lose it to the token pattern.
    assert _strip_boilerplate('He died in 1988. "About the title" may belong.') == "He died in 1988."


def test_urls_in_prose_are_masked(dataset: Path, tmp_path: Path) -> None:
    """Not a leak of the join key, but "no URL anywhere" is a testable invariant."""
    write_jsonl(dataset / "seller.jsonl", [
        {"seller_id": "abebooks:3207104", "seller_name": "Book Lover's Warehouse",
         "seller_description": "See (http://www.example.com) or www.example.org for details."},
    ])
    run(dataset, tmp_path)
    blob = (tmp_path / "lake" / "source_tables.jsonl").read_text(encoding="utf-8")
    assert "http" not in blob and "www." not in blob
    assert "[link]" in blob


def test_an_isbn_quoted_in_a_synopsis_is_masked(dataset: Path, tmp_path: Path) -> None:
    run(dataset, tmp_path)
    table = read_lake(tmp_path / "lake")[0]
    index = [c["column_name"] for c in table["columns"]].index("synopsis_text")
    assert table["rows"][0]["cells"][index]["text"] == "Classic text. ISBN [ISBN] on the back."


def test_each_table_declares_its_name_column_as_the_entity(dataset: Path, tmp_path: Path) -> None:
    """Without this the existing pipeline yields zero query views.

    ``column_profiles`` decides entity columns by Wikipedia link density, which
    is always zero for scraped data, so the index is set explicitly instead.
    """
    run(dataset, tmp_path)
    tables = read_lake(tmp_path / "lake")
    for table, expected in zip(tables, ("title", "seller_name")):
        names = [c["column_name"] for c in table["columns"]]
        entity_index = table["metadata"]["candidate_entity_columns"]
        assert [names[index] for index in entity_index] == [expected]


def test_the_built_table_yields_query_views(dataset: Path, tmp_path: Path) -> None:
    """The integration that matters: the built table is consumable as-is."""
    run(dataset, tmp_path)
    table = read_lake(tmp_path / "lake")[0]
    views = generate_query_views(table, max_views=5, seed=1)
    assert views, "the built table produced no query views"
    assert all("title" in view["selected_column_names"] for view in views)
    assert all(view["hidden_column_names"] for view in views)


def test_page_snapshots_are_not_bridge_assets(dataset: Path, tmp_path: Path) -> None:
    run(dataset, tmp_path)
    assets = [json.loads(line) for line in
              (tmp_path / "lake" / "bridge_assets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {asset["asset_id"] for asset in assets} == {"ev:img", "ev:txt", "ev:policy"}
    assert all(asset["url"] is None for asset in assets)


def test_an_image_asset_without_a_download_has_no_local_path(dataset: Path, tmp_path: Path) -> None:
    stats = run(dataset, tmp_path)
    assets = [json.loads(line) for line in
              (tmp_path / "lake" / "bridge_assets.jsonl").read_text(encoding="utf-8").splitlines()]
    image = next(asset for asset in assets if asset["asset_type"] == "image")
    assert image["local_path"] is None and image["sha256"] is None
    assert stats["images_missing"] == 1 and stats["images_resolved"] == 0


def test_an_image_asset_resolves_to_its_file_when_downloaded(dataset: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "image_manifest.jsonl"
    write_jsonl(manifest, [{
        "url": "https://pictures.abebooks.com/inventory/31595123263.jpg", "status": "ok",
        "sha256": "a" * 64, "mime_type": "image/jpeg", "width": 300, "height": 287,
        "local_path": "/cache/images/" + "a" * 64 + ".jpg",
        "relative_path": "images/" + "a" * 64 + ".jpg",
    }])
    args = parser().parse_args(["--data-dir", str(dataset), "--output-dir", str(tmp_path / "lake"),
                                "--image-manifest", str(manifest)])
    stats = build(args)
    assets = [json.loads(line) for line in
              (tmp_path / "lake" / "bridge_assets.jsonl").read_text(encoding="utf-8").splitlines()]
    image = next(asset for asset in assets if asset["asset_type"] == "image")
    assert image["local_path"].endswith("a" * 64 + ".jpg") and image["width"] == 300
    assert stats["images_resolved"] == 1


def test_chunks_are_balanced_and_lose_no_rows() -> None:
    rows = [{"_row_id": f"bk_{n:04d}"} for n in range(1, 254)]
    parts = chunk(rows, 84)
    assert len(parts) == 84
    assert sum(len(part) for part in parts) == 253
    assert {len(part) for part in parts} == {3, 4}


def test_splitting_stops_at_one_row_per_table() -> None:
    assert chunk([{"_row_id": "a"}, {"_row_id": "b"}], 10) == [[{"_row_id": "a"}], [{"_row_id": "b"}]]


def test_table_counts_land_near_the_request() -> None:
    counts = piece_counts(303, 100, {"book": 253, "seller": 50})
    assert sum(counts.values()) == 100
    assert counts["book"] > counts["seller"]     # apportioned by row count


def test_the_split_never_falls_below_the_row_floor() -> None:
    """Asking for 100 tables from 253 books would make 84 tables of 3 rows.

    Those are not a larger lake, they are unusable: the joinability builder's
    ``min_target_rows`` gate rejects them outright, so each would yield no query
    at all.  The floor caps the request instead.
    """
    counts = piece_counts(303, 100, {"book": 253, "seller": 50}, min_rows=5)
    assert sum(counts.values()) == 60
    assert counts == {"book": 50, "seller": 10}
    # A request the floor does not bind is honoured unchanged.
    assert sum(piece_counts(303, 60, {"book": 253, "seller": 50}, min_rows=5).values()) == 60


def test_the_run_reports_the_table_count_it_promised(dataset: Path, tmp_path: Path) -> None:
    stats = run(dataset, tmp_path, "--target-tables", "2")
    assert stats["tables"] == 2 and stats["tables_by_source"] == {"book": 1, "seller": 1}
    assert stats["rows"] == {"book": 1, "seller": 1}


def bridge(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in
            (tmp_path / "lake" / "bridge_assets.jsonl").read_text(encoding="utf-8").splitlines()]


def test_policy_text_lands_on_seller_rows(dataset: Path, tmp_path: Path) -> None:
    """Seller policy describes the seller, so it hangs off the seller row.

    Keyed on ``book_id`` it lands on whichever book happened to quote it -- which
    is where it went before, and it left the seller tables with no text evidence
    at all (46/50 sellers had policy text that never reached their own row).
    """
    run(dataset, tmp_path)
    policy = [a for a in bridge(tmp_path) if a["source"] == "abebooks_seller_policy"]
    assert policy, "the fixture's seller_policy evidence did not survive the build"
    assert all(a["row_id"].startswith("sl_") for a in policy)
    assert not any(a["row_id"].startswith("bk_") for a in policy)
    assert all(a["source_column"] == "terms_of_sale" for a in policy)


def test_every_text_asset_names_the_column_it_was_derived_from(
        dataset: Path, tmp_path: Path) -> None:
    """Guards against running the joinability builder on a stale lake.

    A text asset is normally a verbatim copy of its source column, so without
    this field the builder cannot tell a recovery from a lookup and scores the
    self-pair at a meaningless 100%.
    """
    run(dataset, tmp_path)
    for asset in bridge(tmp_path):
        if asset["asset_type"] == "text":
            assert asset.get("source_column"), asset
        else:
            # A cover photograph has no source column: it can be read for any
            # attribute of the book it depicts, so no pair is a lookup.
            assert asset.get("source_column") is None, asset


def test_rebuilding_is_deterministic(dataset: Path, tmp_path: Path) -> None:
    run(dataset, tmp_path)
    first = (tmp_path / "lake" / "source_tables.jsonl").read_text(encoding="utf-8")
    run(dataset, tmp_path)
    assert (tmp_path / "lake" / "source_tables.jsonl").read_text(encoding="utf-8") == first

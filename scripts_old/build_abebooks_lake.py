#!/usr/bin/env python
"""Turn the four scraped tables into a two-table multimodal lake.

The four tables (``book_edition`` / ``seller`` / ``book_listing`` /
``evidence_asset``) are a *star schema*: every join key is visible, so every
join is trivial.  A joinability benchmark needs the opposite -- tables whose
entities can only be matched through the text and image evidence hanging off
them.  So this builder makes three deliberate breaks with the scrape:

**The entity is the name, not the identifier.**  ``title`` and ``seller_name``
become the entity columns; ``book_id`` / ``isbn10`` / ``isbn13`` / ``seller_id``
go.  A cover photograph carries the title and author but not the ISBN (verified
against a downloaded cover), so the name is the column a model can actually
recover from the pixels.

**Identifiers ride along in places that are not identifier columns.**  Removing
``isbn13`` is not enough.  ``seller_inventory_no`` *is* the ISBN for 130/253
rows (``7719-9780201632088``); the detail-page URL spells out a title/author
slug; page snapshots are named after the ISBN on disk.  Every URL column is
dropped here and image files are keyed by content hash instead.

**Rows are keyed positionally.**  ``row_id`` values are ``bk_0001``-style,
assigned after a seeded shuffle, so a row's key cannot be brute-forced back
into the ISBN it replaced.  The mapping to the original ids lives in
``key_map.jsonl``, which is audit material and never a model input.

This lives in ``scripts_old`` rather than ``src`` because it is AbeBooks-native:
the lake's entity model (name-as-entity, positional row ids, no Wikipedia layer)
is not the one the shared ``mmdd_dataset`` builders assume, and the shared
modules should not grow a second provider branch to host it.

Usage::

    python scripts_old/build_abebooks_lake.py
    python scripts_old/build_abebooks_lake.py --target-tables 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Running this file directly puts ``scripts_old`` on ``sys.path``, not ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmdd_dataset.utils import clean_text, read_jsonl, sanitize_cell_text, write_json, write_jsonl

#: AbeBooks appends ``<isbn><AB|B><MMDDYY> "About the title" may belong to
#: another edition of this title.`` to bios.  It is both a leak (the ISBN, in
#: plain sight) and junk (it is not part of the author's biography).
#:
#: The notice is matched *unanchored and open-ended* rather than as a whole
#: literal: ``build_abebooks_dataset.py`` caps cells at 1024 characters, which
#: cuts the notice off mid-phrase for some rows (``... "About the title'``).
#: A ``$``-anchored literal silently misses those and leaves the ISBN behind.
BOILERPLATE_MARKER = '"About the title'
#: The identifier token AbeBooks writes immediately before the notice:
#: ``0201756080AB06262002`` -- the ISBN-10, a one-to-three letter code, and an
#: 8-digit date.  Stated as a shape rather than by enumerating the 175 observed
#: spellings, since the letters vary (``AB`` 84x, ``B`` 4x, others 5x) and the
#: ISBN is always the 10-digit form.  ``(?:...)?`` because 91/175 notices carry
#: no token at all, where the pattern matches the empty string and changes
#: nothing.
TRAILING_ID = re.compile(r"[\s,]*(?:[0-9Xx]{10,13}[A-Za-z]{1,3}\d{8})?[\s,]*$")
#: Any URL in prose.  Stopping short of the closing punctuation keeps the
#: surrounding sentence intact (``(http://x.com).`` -> ``([link]).``).
URL_IN_TEXT = re.compile(r"\b(?:https?://|www\.)[^\s,;)\]}<>\"']+", re.IGNORECASE)

#: Columns carried from ``book_edition``.  Chosen as "describes the book", with
#: identifiers, provenance and URLs removed (see the module docstring).
BOOK_EDITION_COLUMNS = (
    "title", "authors", "publisher", "publication_year", "publication_year_raw",
    "language", "binding", "edition_number", "series", "dust_jacket", "product_type",
    "dimensions", "item_weight", "copy_condition_grade", "catalogue_image_kind",
    "synopsis_text", "about_author_text", "goodreads_rating", "goodreads_rating_count",
)
#: Offer columns folded in from ``book_listing`` -- one row per book is the
#: representative offer, so these are book attributes at this grain.  Dropped:
#: ``listing_url``/``vendor_image_url`` (URLs), ``seller_inventory_no`` (is the
#: ISBN for 130 rows), ``listing_title``/``listing_authors`` (byte-identical
#: duplicates of ``title``/``authors``).
BOOK_LISTING_COLUMNS = (
    "condition", "condition_description", "availability_quantity", "price", "currency",
    "shipping_price", "shipping_currency", "total_price", "vendor_description",
    "edition_marker", "stock_image_flag", "image_kind",
)
#: Dropped from ``seller``: ``seller_id``/``seller_url`` (identifier and URL) and
#: ``seller_name_variants`` -- an alias list for the very column we are asking a
#: model to recover, which would hand over the answer.
SELLER_COLUMNS = (
    "seller_name", "location_raw", "city", "region", "country", "seller_rating",
    "seller_rating_kind", "seller_since", "seller_since_iso", "specialties",
    "terms_of_sale", "shipping_terms", "seller_description", "seller_name_collision",
)

#: Evidence worth carrying as a bridge asset.  The page snapshots are excluded:
#: they are whole HTML documents, useful for build audit but not as material a
#: model can read a value out of.
TEXT_ASSETS = ("description", "synopsis", "about_author", "seller_policy", "shipping_policy")
IMAGE_ASSETS = ("catalogue_cover", "seller_cover")

#: Which table column each text asset was *derived from*.  This is not
#: bookkeeping: a text asset is usually a verbatim copy of its source column, so
#: "recover `synopsis_text` from the `synopsis` asset" is a lookup, not a
#: recovery, and would report a perfect score for doing nothing.  The joinability
#: generator needs this to refuse the self-pair.  ``None`` for an asset type with
#: no source column.
TEXT_SOURCE_COLUMNS = {
    "description": "vendor_description",
    "synopsis": "synopsis_text",
    "about_author": "about_author_text",
    "seller_policy": "terms_of_sale",
    "shipping_policy": "shipping_terms",
}

#: Asset types that describe the *seller* rather than the book, and so hang off
#: a seller row.  Their evidence records carry a ``listing_id``; the seller is
#: reached through ``book_listing.listing_id -> seller_id``.  ``seller_cover``
#: is deliberately not here: it is a photograph of one specific book, and
#: attaching it to a five-row seller table would claim it depicts all five.
SELLER_ASSETS = ("seller_policy", "shipping_policy")

#: Text columns that may quote an ISBN and must not.
SCRUBBED_TEXT_COLUMNS = ("synopsis_text", "vendor_description", "about_author_text")


def _strip_boilerplate(value: str | None) -> str | None:
    """Drop AbeBooks' trailing "may belong to another edition" notice.

    Cut from the notice onward, then drop the ``<isbn><code>`` token that sits
    immediately before it.  Order matters: the token is only recognisable once
    the notice is gone, and the row it belongs to may be truncated.
    """
    if not value:
        return value
    at = value.find(BOILERPLATE_MARKER)
    if at == -1:
        return value
    cleaned = TRAILING_ID.sub("", value[:at]).strip()
    return cleaned or None


def _mask_urls(value: str | None) -> str | None:
    return URL_IN_TEXT.sub("[link]", value) if value else value


def _mask_isbn(value: str | None, isbns: list[str]) -> str | None:
    """Replace any ISBN the text quotes with a placeholder.

    Only 5/253 synopses do this, but a model that can read the join key out of a
    synopsis never has to look at the cover, which defeats the point.
    """
    if not value:
        return value
    out = value
    for isbn in isbns:
        if isbn and isbn in out:
            out = out.replace(isbn, "[ISBN]")
    return out


def book_rows(editions: list[dict], listings: list[dict], max_chars: int) -> list[dict]:
    """One row per book: bibliographic columns with its representative offer folded in."""
    offer_of = {row["book_id"]: row for row in listings}
    rows: list[dict] = []
    for edition in editions:
        offer = offer_of.get(edition["book_id"], {})
        row: dict[str, Any] = {}
        for column in BOOK_EDITION_COLUMNS:
            row[column] = edition.get(column)
        for column in BOOK_LISTING_COLUMNS:
            row[column] = offer.get(column)
        for column in SCRUBBED_TEXT_COLUMNS:
            if column in row:
                row[column] = _mask_isbn(_strip_boilerplate(row[column]),
                                         [edition.get("isbn10"), edition.get("isbn13")])
        for column, value in list(row.items()):
            if isinstance(value, str):
                row[column] = _mask_urls(sanitize_cell_text(value, max_chars)) or None
        row["_source_key"] = edition["book_id"]
        rows.append(row)
    return rows


def seller_rows(sellers: list[dict], max_chars: int) -> list[dict]:
    rows: list[dict] = []
    for seller in sellers:
        row = {column: seller.get(column) for column in SELLER_COLUMNS}
        for column, value in list(row.items()):
            if isinstance(value, str):
                row[column] = _mask_urls(sanitize_cell_text(value, max_chars)) or None
        row["_source_key"] = seller["seller_id"]
        rows.append(row)
    return rows


def assign_row_ids(rows: list[dict], prefix: str, seed: int) -> list[dict]:
    """Relabel rows ``<prefix>_0001``... after a seeded shuffle.

    Positional keys are the point: a key derived from the ISBN would be
    recoverable by enumerating the ISBN space and hashing.
    """
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    out: list[dict] = []
    for position, index in enumerate(order, 1):
        row = dict(rows[index])
        row["_row_id"] = f"{prefix}_{position:04d}"
        out.append(row)
    return out


def chunk(rows: list[dict], pieces: int) -> list[list[dict]]:
    """Split into ``pieces`` contiguous chunks whose sizes differ by at most one."""
    if pieces <= 0 or not rows:
        return []
    pieces = min(pieces, len(rows))
    base, extra = divmod(len(rows), pieces)
    out: list[list[dict]] = []
    start = 0
    for index in range(pieces):
        size = base + (1 if index < extra else 0)
        out.append(rows[start:start + size])
        start += size
    return out


def piece_counts(total: int, target: int, sizes: dict[str, int],
                 min_rows: int = 1) -> dict[str, int]:
    """Apportion ``target`` tables across sources in proportion to their row counts.

    ``min_rows`` caps the split: a table below the joinability row floor is not a
    smaller lake, it is an unusable one (``min_target_rows`` rejects it, so the
    table yields no query at all).  Asking for 100 tables from 253 books therefore
    yields 50, not 84 tables of 3 rows that no downstream stage can use.  The
    realised count is reported in ``stats.json`` next to the requested one.
    """
    if target <= 0:
        return {name: 1 for name in sizes}
    out: dict[str, int] = {}
    for name, rows in sizes.items():
        ceiling = max(1, rows // min_rows) if min_rows > 0 else rows
        out[name] = max(1, min(rows, ceiling, round(target * rows / total)))
    return out


def as_source_table(name: str, table_id: str, columns: list[str], rows: list[dict],
                    entity_column: str) -> dict[str, Any]:
    """One lake table, in the shape ``mmdd_dataset.tables`` produces.

    ``candidate_entity_columns`` is set directly rather than inferred: the
    existing detector keys off Wikipedia link density (``column_profiles``), which
    is always zero here, so an inferred table would yield no query views at all.
    """
    entity_index = columns.index(entity_column)
    cells = [
        [
            {
                "column_index": index,
                "column_name": column,
                "raw": row.get(column),
                "text": "" if row.get(column) is None else str(row.get(column)),
                "wiki_title": None,
                "has_wiki_link": False,
            }
            for index, column in enumerate(columns)
        ]
        for row in rows
    ]
    profiles = []
    for index, column in enumerate(columns):
        values = [row.get(column) for row in rows]
        non_empty = [value for value in values if value not in (None, "")]
        profiles.append({
            "column_index": index,
            "non_empty_ratio": round(len(non_empty) / max(1, len(rows)), 6),
            "wiki_link_ratio": 0.0,
            "unique_ratio": round(len(set(map(str, non_empty))) / max(1, len(non_empty)), 6),
            "numeric_ratio": 0.0,
        })
    return {
        "source_table_id": table_id,
        "source_name": name,
        "num_rows": len(rows),
        "num_cols": len(columns),
        "columns": [{"column_index": index, "column_name": column}
                    for index, column in enumerate(columns)],
        "rows": [{"row_id": row["_row_id"], "cells": row_cells}
                 for row, row_cells in zip(rows, cells)],
        "metadata": {"column_profiles": profiles, "candidate_entity_columns": [entity_index]},
    }


def bridge_assets(evidence: list[dict], images: dict[str, dict], key_of: dict[str, str],
                  seller_of_listing: dict[str, str], seller_key_of: dict[str, str],
                  max_chars: int) -> list[dict]:
    """Evidence rows as bridge assets, with downloaded images resolved to files.

    Book evidence resolves through ``book_id``; seller policy resolves through
    ``listing_id -> seller_id`` so it lands on the seller it describes rather
    than on whichever book happened to quote it.
    """
    assets: list[dict] = []
    for record in evidence:
        asset_type = record.get("asset_type")
        if asset_type in SELLER_ASSETS:
            seller_id = seller_of_listing.get(record.get("listing_id"))
            row_id = seller_key_of.get(seller_id) if seller_id else None
        else:
            row_id = key_of.get(record.get("book_id"))
        if row_id is None:
            continue
        if asset_type in TEXT_ASSETS:
            content = _strip_boilerplate(record.get("extracted_text"))
            content = _mask_urls(_mask_isbn(content, [record.get("record_isbn")]))
            assets.append({
                "asset_id": record["evidence_id"],
                "asset_type": "text",
                "source": f"abebooks_{asset_type}",
                "source_column": TEXT_SOURCE_COLUMNS.get(asset_type),
                "row_id": row_id,
                "content": sanitize_cell_text(content, max_chars) if content else None,
                "url": None,
                "local_path": None,
            })
        elif asset_type in IMAGE_ASSETS:
            fetched = images.get(record.get("uri")) or {}
            ok = fetched.get("status") == "ok"
            assets.append({
                "asset_id": record["evidence_id"],
                "asset_type": "image",
                "source": f"abebooks_{asset_type}",
                # An image has no source column: a cover can be read for any
                # attribute of the book it depicts, so no pair is a lookup.
                "source_column": None,
                "row_id": row_id,
                "content": None,
                "url": None,
                "local_path": fetched.get("local_path") if ok else None,
                "relative_path": fetched.get("relative_path") if ok else None,
                "sha256": fetched.get("sha256") if ok else None,
                "mime_type": fetched.get("mime_type") if ok else None,
                "width": fetched.get("width") if ok else None,
                "height": fetched.get("height") if ok else None,
            })
    return assets


def build(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load = lambda name: list(read_jsonl(data_dir / f"{name}.jsonl"))  # noqa: E731
    editions, sellers, listings = load("book_edition"), load("seller"), load("book_listing")
    evidence = load("evidence_asset")

    books = assign_row_ids(book_rows(editions, listings, args.max_cell_chars), "bk", args.seed)
    seller_records = assign_row_ids(seller_rows(sellers, args.max_cell_chars), "sl", args.seed)

    counts = piece_counts(len(books) + len(seller_records), args.target_tables,
                          {"book": len(books), "seller": len(seller_records)},
                          args.min_rows_per_table)
    tables: list[dict] = []
    for name, rows, columns, entity in (
        ("book", books, list(BOOK_EDITION_COLUMNS) + list(BOOK_LISTING_COLUMNS), "title"),
        ("seller", seller_records, list(SELLER_COLUMNS), "seller_name"),
    ):
        for index, part in enumerate(chunk(rows, counts[name]), 1):
            tables.append(as_source_table(name, f"st_{name}_{index:03d}", columns, part, entity))

    manifest_path = Path(args.image_manifest)
    images = ({row["url"]: row for row in read_jsonl(manifest_path)}
              if manifest_path.exists() else {})
    key_of = {row["_source_key"]: row["_row_id"] for row in books}
    seller_key_of = {row["_source_key"]: row["_row_id"] for row in seller_records}
    seller_of_listing = {row["listing_id"]: row["seller_id"] for row in listings}
    assets = bridge_assets(evidence, images, key_of, seller_of_listing, seller_key_of,
                           args.max_cell_chars)

    write_jsonl(out_dir / "source_tables.jsonl", tables)
    write_jsonl(out_dir / "bridge_assets.jsonl", assets)
    write_jsonl(out_dir / "key_map.jsonl", (
        [{"row_id": row["_row_id"], "table": "book", "source_key": row["_source_key"]}
         for row in books] +
        [{"row_id": row["_row_id"], "table": "seller", "source_key": row["_source_key"]}
         for row in seller_records]
    ))

    by_type: dict[str, int] = {}
    for asset in assets:
        by_type[asset["asset_type"]] = by_type.get(asset["asset_type"], 0) + 1
    by_family: dict[str, int] = {}
    for asset in assets:
        by_family[asset["source"]] = by_family.get(asset["source"], 0) + 1
    text_without_source_column = sum(
        1 for asset in assets
        if asset["asset_type"] == "text" and not asset.get("source_column"))
    stats = {
        "tables": len(tables),
        "tables_by_source": counts,
        "target_tables_requested": args.target_tables,
        "min_rows_per_table": args.min_rows_per_table,
        "rows": {"book": len(books), "seller": len(seller_records)},
        "rows_per_table": {
            "min": min(table["num_rows"] for table in tables),
            "max": max(table["num_rows"] for table in tables),
        },
        "columns": {"book": tables[0]["num_cols"], "seller": tables[-1]["num_cols"]},
        "bridge_assets": by_type,
        "bridge_assets_by_family": by_family,
        "text_assets_without_source_column": text_without_source_column,
        "assets_by_row_source": {
            name: sum(1 for asset in assets
                      if asset["row_id"].startswith("bk" if name == "book" else "sl"))
            for name in ("book", "seller")
        },
        "images_resolved": sum(1 for asset in assets
                               if asset["asset_type"] == "image" and asset["local_path"]),
        "images_missing": sum(1 for asset in assets
                              if asset["asset_type"] == "image" and not asset["local_path"]),
    }
    if text_without_source_column:
        raise SystemExit(
            f"{text_without_source_column} text assets have no source_column; the "
            "joinability builder refuses to run without it (it cannot tell a "
            "recovery from a lookup)")
    write_json(out_dir / "stats.json", stats)
    write_json(out_dir / "lake_manifest.json", {
        "format": "mmdd_abebooks_lake_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "seed": args.seed,
        "target_tables": args.target_tables,
        "inputs": {name: {"path": str(data_dir / f"{name}.jsonl"),
                          "sha256": _sha256(data_dir / f"{name}.jsonl"),
                          "records": len(rows)}
                   for name, rows in (("book_edition", editions), ("seller", sellers),
                                      ("book_listing", listings), ("evidence_asset", evidence))},
        "stats": stats,
    })

    print(f"{stats['tables']} tables ({counts['book']} book + {counts['seller']} seller), "
          f"{stats['rows_per_table']['min']}-{stats['rows_per_table']['max']} rows each")
    print(f"  book {stats['columns']['book']} cols, seller {stats['columns']['seller']} cols")
    print(f"  bridge assets {by_type}  images resolved {stats['images_resolved']}/"
          f"{stats['images_resolved'] + stats['images_missing']}")
    print(f"  {out_dir / 'source_tables.jsonl'}")
    return stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--data-dir", default="output/abebooks_dataset")
    result.add_argument("--image-manifest", default="output/abebooks_images/image_manifest.jsonl")
    result.add_argument("--output-dir", default="output/abebooks_lake")
    result.add_argument("--target-tables", type=int, default=60,
                        help="how many tables to split the lake into, in total")
    result.add_argument("--min-rows-per-table", type=int, default=5,
                        help="never split below this many rows; the joinability "
                             "builder's min_target_rows gate rejects smaller tables")
    result.add_argument("--max-cell-chars", type=int, default=4096)
    result.add_argument("--seed", type=int, default=13)
    return result


def main(argv: list[str] | None = None) -> int:
    build(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

import numpy
from PIL import Image

# Running this file directly puts ``scripts_old`` on ``sys.path``, not ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmdd_dataset.utils import clean_text, read_jsonl, sanitize_cell_text, write_json, write_jsonl

#: Every seller of one book shows the same cover artwork, so "are these the same
#: picture?" cannot be answered by looking at the artwork -- it has to be
#: answered by looking at the *image*.  A content hash misses the re-encodes:
#: one seller uploads the cover at 287x300 and another at 288x300, and the two
#: share nothing but the pixels.  Measured against the 2529 downloaded covers, a
#: 64x64 RGB comparison that tolerates a small translation separates them
#: cleanly -- lookalikes sit at or below 5, while everything from ~20 up is a
#: *photograph of the physical copy* set against the flat artwork, which is the
#: evidence that reads condition and binding, so folding those would destroy
#: the very signal the image channel exists to carry.
IMAGE_GRID = 64
IMAGE_SHIFT = 4
IMAGE_DUPLICATE_MAX_DISTANCE = 5.0

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

#: Columns that are never emitted into a table, and why each is absent.
#:
#: Three separate defects collapse into one fix, because all three are the same
#: thing: the column's value is already sitting in the row's evidence or in
#: another column, so "recovering" it asks a model to copy something it can see.
#: Deleting the column removes the possibility; the alternative was teaching the
#: shared builder three guards it has no notion of, for a corpus whose assets are
#: column copies rather than documents.
#:
#: ``evidence_copy`` -- the text asset *is* this column, verbatim (its
#: ``source_column``; 1141/1141/1099/100/94 assets each).  Asking for the column
#: from that asset is a lookup that scores 100% without reading anything.
#:
#: ``contained_in_another_column`` -- measured on the 130-table lake with the
#: same judgement the copy-channel detector uses (casefold, either side contains
#: the other, contained side at least 4 characters; the floor matters, or short
#: numbers make every numeric pair look like a match).  In each pair the survivor
#: is the one the external review endorsed more often: ``copy_condition_grade``
#: 69% against ``condition`` 49%, ``publisher`` 45% against ``series``, and
#: ``location_raw`` against ``city``/``country`` on the seller side.
#:
#: ``describes_the_asset`` -- image metadata (is this a stock cover or a seller's
#: own photo).  Recovering it from the photograph is circular.
EXCLUDED_COLUMNS = {
    "about_author_text": "evidence_copy",
    "synopsis_text": "evidence_copy",
    "vendor_description": "evidence_copy",
    "terms_of_sale": "evidence_copy",
    "shipping_terms": "evidence_copy",
    "condition": "contained_in_another_column",
    "publication_year_raw": "contained_in_another_column",
    "series": "contained_in_another_column",
    "city": "contained_in_another_column",
    "country": "contained_in_another_column",
    "image_kind": "describes_the_asset",
    "catalogue_image_kind": "describes_the_asset",
    "stock_image_flag": "describes_the_asset",
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


def _duplicate_key(asset: dict) -> tuple | None:
    """Identity of an asset's payload, or None when it must never be folded.

    An image whose download failed carries no sha, so every such asset would
    share the key ``(None)`` -- keep them all rather than collapse unrelated
    failures into one row.
    """
    if asset["asset_type"] == "image":
        sha = asset.get("sha256")
        return ("image", asset["row_id"], sha) if sha else None
    return ("text", asset["row_id"], asset["source"], asset["content"])


def _thumbnail(path: Path) -> Any | None:
    """A ``IMAGE_GRID`` square RGB thumbnail, or None if the file is unreadable."""
    try:
        with Image.open(path) as handle:
            resized = handle.convert("RGB").resize((IMAGE_GRID, IMAGE_GRID), Image.LANCZOS)
    except (OSError, ValueError):
        return None
    return numpy.asarray(resized, dtype=numpy.float32)


def _image_distance(left: Any, right: Any) -> float:
    """Mean absolute RGB difference, minimised over small translations.

    The shift search is what makes a re-encode comparable at all: two uploads
    of one photograph are rarely cropped to the same box, and a one-pixel
    offset on a 64x64 grid swamps the difference between "the same photo" and
    "a different photo".  The border is excluded from the comparison so the
    pixels rolled in at the edge cannot pass themselves off as content.
    """
    inner = (slice(IMAGE_SHIFT, IMAGE_GRID - IMAGE_SHIFT),
             slice(IMAGE_SHIFT, IMAGE_GRID - IMAGE_SHIFT))
    best = float("inf")
    for dy in range(-IMAGE_SHIFT, IMAGE_SHIFT + 1):
        for dx in range(-IMAGE_SHIFT, IMAGE_SHIFT + 1):
            moved = numpy.roll(numpy.roll(right, dy, axis=0), dx, axis=1)
            distance = float(numpy.abs(left[inner] - moved[inner]).mean())
            if distance < best:
                best = distance
    return best


def _lookalike_ids(images: list[tuple[dict, Any]], max_distance: float) -> set[str]:
    """Asset ids to drop: all but one per single-linkage cluster of lookalikes."""
    parent = list(range(len(images)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if _image_distance(images[i][1], images[j][1]) <= max_distance:
                left, right = find(i), find(j)
                if left != right:
                    parent[max(left, right)] = min(left, right)

    clusters: dict[int, list[dict]] = {}
    for index, (asset, _) in enumerate(images):
        clusters.setdefault(find(index), []).append(asset)
    dropped: set[str] = set()
    for members in clusters.values():
        survivors = sorted(members, key=lambda asset: asset["asset_id"])
        dropped.update(asset["asset_id"] for asset in survivors[1:])
    return dropped


def dedupe_assets(assets: list[dict], image_root: Path | str | None = None,
                  max_distance: float = IMAGE_DUPLICATE_MAX_DISTANCE) -> list[dict]:
    """Drop repeated evidence within a row, keeping the lowest asset_id.

    A book's detail page carries one panel per listing, so the same vendor
    blurb and the same cover photograph arrive once per listing. Identical
    evidence recovers nothing a second time, it only multiplies the extraction
    budget, and counting it as several independent recoveries would flatter
    the recovery gate. Keeping the lowest asset_id makes the survivor
    deterministic across rebuilds.

    Text folds on exact content, which is the whole of it -- a blurb is either
    byte-identical to its sibling or it says something else. Images need the
    pixel comparison as well, because the same photograph reaches the lake
    re-encoded at a different size under half a dozen listing urls; pass
    ``image_root`` to resolve ``local_path`` against and those fold too. An
    image whose file is missing is never folded: without the pixels there is
    no evidence the two are the same picture, and guessing would be worse
    than keeping the extra row.
    """
    positions: dict[tuple, int] = {}
    kept: list[dict] = []
    for asset in assets:
        key = _duplicate_key(asset)
        if key is None:
            kept.append(asset)
            continue
        at = positions.get(key)
        if at is None:
            positions[key] = len(kept)
            kept.append(asset)
        elif asset["asset_id"] < kept[at]["asset_id"]:
            kept[at] = asset

    if image_root is None or max_distance <= 0:
        return kept

    by_row: dict[str, list[tuple[dict, Any]]] = {}
    for asset in kept:
        local = asset.get("local_path")
        if asset["asset_type"] != "image" or not local:
            continue
        thumbnail = _thumbnail(Path(image_root) / local)
        if thumbnail is not None:
            by_row.setdefault(asset["row_id"], []).append((asset, thumbnail))

    dropped: set[str] = set()
    for group in by_row.values():
        if len(group) > 1:
            dropped |= _lookalike_ids(group, max_distance)
    return [asset for asset in kept if asset["asset_id"] not in dropped]


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
        # The exclusion is applied here rather than downstream so no cell, profile
        # or asset ever refers to the column: an absent column cannot be offered
        # as a hidden join key, cannot be shown as context, and cannot be the
        # answer a model copies out of the evidence that contains it.
        if entity in EXCLUDED_COLUMNS:
            raise SystemExit(
                f"{entity!r} is the entity column for {name} tables and cannot be "
                "excluded; every table would lose its identity")
        kept = [column for column in columns if column not in EXCLUDED_COLUMNS]
        if not kept:
            raise SystemExit(f"excluding columns left the {name} table with none")
        for index, part in enumerate(chunk(rows, counts[name]), 1):
            tables.append(as_source_table(name, f"st_{name}_{index:03d}", kept, part, entity))

    manifest_path = Path(args.image_manifest)
    images = ({row["url"]: row for row in read_jsonl(manifest_path)}
              if manifest_path.exists() else {})
    key_of = {row["_source_key"]: row["_row_id"] for row in books}
    seller_key_of = {row["_source_key"]: row["_row_id"] for row in seller_records}
    seller_of_listing = {row["listing_id"]: row["seller_id"] for row in listings}
    raw_assets = bridge_assets(evidence, images, key_of, seller_of_listing, seller_key_of,
                               args.max_cell_chars)
    exact = dedupe_assets(raw_assets)
    assets = dedupe_assets(raw_assets, args.image_root, args.image_max_distance)
    assets_dropped = len(raw_assets) - len(assets)

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
        "excluded_columns": dict(sorted(EXCLUDED_COLUMNS.items())),
        "bridge_assets": by_type,
        "bridge_assets_by_family": by_family,
        "bridge_assets_before_dedupe": len(raw_assets),
        "bridge_assets_deduped": assets_dropped,
        "image_lookalikes_merged": len(exact) - len(assets),
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
    print(f"  dropped {assets_dropped} duplicate assets "
          f"({len(raw_assets)} -> {len(assets)}; one per (row, payload))")
    print(f"    of which {len(exact) - len(assets)} were re-encoded covers of a "
          f"picture already in the row")
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
    result.add_argument("--image-root", default=".",
                        help="root the manifest's relative local_path values resolve "
                             "against, for the lookalike-cover comparison")
    result.add_argument("--image-max-distance", type=float,
                        default=IMAGE_DUPLICATE_MAX_DISTANCE,
                        help="fold two covers in one row when their thumbnails differ "
                             "by less than this; 0 disables the pixel comparison and "
                             "leaves only the exact-content fold")
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

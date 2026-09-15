"""Build the four AbeBooks tables from collected scrape records.

``src/abebooks_scraper.py`` owns ``output/abebooks_full.jsonl``; this module only
reads it.  Its job is to turn per-ISBN search + detail records into the four
physical tables the plan doc specifies, with every derived value attributable to
a page, a selector and a timestamp.

Two things the data forced, both measured rather than assumed:

* The detail page's ``book`` object describes exactly one listing, and 12 of the
  13 card keys are non-null on every card.  "Structured field completeness"
  therefore cannot discriminate between candidate cards, so selection reduces to
  one question -- anchor on the listing that has the detail page, or on the card
  with the nicer image.  We anchor on the detail listing, because 19 columns
  exist only for it and collection is paused, so swapping the pick would null
  them permanently.  ``prefer_seller_image`` keeps the other answer measurable.
* ``book.title``/``book.authors`` are byte-identical to the chosen card's
  strings, so they are one seller's claim, not an edition truth.  The columns are
  ``title``/``authors`` and carry ``*_source``, not ``canonical_*``.

Nothing here touches the network and nothing writes back to the scrape file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mmdd_dataset.utils import clean_text, normalize, sanitize_cell_text, stable_hash

# Statuses ranked best-first, so a later, worse record can never displace a
# better earlier one when the file is re-read after another scraping run.
STATUS_RANK = {"ok": 5, "no_listings": 4, "degraded": 3, "blocked": 2, "error": 1}

# Values the pages use in place of an answer.  They are not data, and leaving
# them in inflates coverage: 36 of the 46 non-null `dimensions` values are the
# literal string "N/A", so the honest coverage is 10/253, not 18%.  Comparison
# goes through `normalize`, which strips punctuation, so the keys below are the
# normalized spellings.
SENTINELS = {"n/a", "na", "n.a.", "not available", "unknown", "none", "null", "-", "--"}
_SENTINEL_KEYS = frozenset(normalize(value) for value in SENTINELS)

# Image families on the CDN.  The URL decides the kind, not the `stock_image`
# flag: 13 cards carry stock_image=False but an /isbn/ URL, and /isbn/ is
# definitionally the publisher cover.  The URL is also what a downloader would
# fetch, so it is the thing to classify.
SELLER_PHOTO_MARKER = "pictures.abebooks.com/inventory/"
STOCK_COVER_MARKER = "pictures.abebooks.com/isbn/"

# Narrow on purpose: words like "paperback" appear in thousands of legitimate
# titles, so flagging them would drown the signal.  These mark a copy that is a
# different edition from the ISBN's own.
EDITION_MARKERS = (
    "international edition", "international student", "global edition",
    "large print", "study guide", "annotated instructor", "instructor's edition",
    "examination copy",
)

# (asset_type, key in the parsed `book`, the selector the value came from)
TEXT_ASSETS = (
    ("description", "vendor_description", "data-test-id='description-text'"),
    ("synopsis", "synopsis_text", "ld+json Product.description"),
    ("about_author", "about_author_text", "data-test-id='about-the-title'"),
    ("seller_policy", "seller_terms", "data-test-id='policy-sales-terms-text'"),
    ("shipping_policy", "shipping_terms", "data-test-id='policy-shipping-terms-text'"),
)

# Controlled vocabulary.  "seller_photo_elsewhere" is the case the pick could not
# express: the detail listing was chosen and its own image is a stock cover, but
# another card for this ISBN carries a real photo.  Without it that majority of
# records would have to be mislabelled as one of the other two.
SELECTED_REASONS = (
    "evidence_complete",
    "seller_photo_elsewhere",
    "stock_image_only",
    "seller_image_preferred",
    "detail_unmatched",
)

MONTHS = {name: index for index, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"), start=1)}


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #

def load_records(path: Path) -> list[dict[str, Any]]:
    """Parsed records in file order, each tagged with its line position.

    The position is the last tiebreak in :func:`dedupe_records` and is stable
    because the scraper only ever appends.
    """
    records: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            record = json.loads(line)
            record["_line_index"] = index
            records.append(record)
    return records


def dedupe_records(records: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """One record per ISBN: best status, then latest, then last in the file.

    This is a pure function of the record set, so re-applying it to the whole
    file after the scraper appends more records yields the same answer for every
    ISBN whose best record did not change -- which is what makes an incremental
    rebuild safe.  Losers are returned rather than dropped so the choice stays
    auditable.
    """
    by_isbn: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for position, record in enumerate(records):
        # load_records stamps the file line; a caller passing bare records in
        # order gets the same ordering from the enumeration.
        order = record.get("_line_index", position)
        by_isbn.setdefault(record.get("isbn") or "", []).append((order, record))

    winners: list[tuple[int, dict[str, Any]]] = []
    superseded: list[dict[str, Any]] = []
    for isbn, group in by_isbn.items():
        ranked = sorted(
            group,
            key=lambda item: (STATUS_RANK.get(item[1].get("status"), 0),
                              item[1].get("retrieved_at") or "",
                              item[0]),
        )
        best_order, best = ranked[-1]
        # A "no_listings" verdict deletes a book from the tables, so require the
        # page to actually show it: a throttled search page also parses to zero
        # cards, and accepting one would silently drop a book we already had.
        if best.get("status") == "no_listings" and not _is_genuine_zero_result(best):
            genuine = [item for item in ranked
                       if item[1].get("status") == "no_listings"
                       and _is_genuine_zero_result(item[1])]
            if genuine:
                best_order, best = genuine[-1]
        winners.append((best_order, best))
        for order, record in group:
            if record is best:
                continue
            superseded.append({
                "isbn": isbn,
                "status": record.get("status"),
                "retrieved_at": record.get("retrieved_at"),
                "line_index": order,
                "reason": f"lost to a {best.get('status')} record for the same ISBN",
            })
    winners.sort(key=lambda item: item[0])
    superseded.sort(key=lambda row: (row["isbn"], row["line_index"]))
    return [record for _, record in winners], superseded


def _is_genuine_zero_result(record: dict[str, Any]) -> bool:
    search = record.get("search")
    if not record.get("html_path") or not record.get("html_sha256") or not search:
        return False
    return (search.get("listing_count") == 0
            and "0 result" in (search.get("result_count_text") or "").lower())


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #

def _isbn13_check_digit(twelve: str) -> str:
    total = sum((1 if position % 2 == 0 else 3) * int(digit)
                for position, digit in enumerate(twelve))
    return str((10 - total % 10) % 10)


def to_isbn13(raw: Any) -> str | None:
    """ISBN-13 for a 10- or 13-digit ISBN, or None when the checksum fails."""
    digits = re.sub(r"[^0-9Xx]", "", str(raw or "")).upper()
    if len(digits) == 13 and digits.isdigit():
        return digits if digits[12] == _isbn13_check_digit(digits[:12]) else None
    if len(digits) != 10 or not digits[:9].isdigit():
        return None
    check = 10 if digits[9] == "X" else int(digits[9]) if digits[9].isdigit() else None
    if check is None:
        return None
    weighted = sum((10 - position) * int(digit) for position, digit in enumerate(digits[:9]))
    if (weighted + check) % 11:
        return None
    twelve = "978" + digits[:9]
    return twelve + _isbn13_check_digit(twelve)


def book_id_for(raw_isbn: Any) -> tuple[str | None, str]:
    """``book_id`` and the scheme that produced it.

    ISBN-13 is the key form because the seed and the page agree on it 253/253,
    while a 10-digit seed would key differently from the same book's 13-digit
    page value.  An ISBN whose checksum does not validate is kept under an
    explicit ``isbn10_fallback`` scheme rather than dropped, so a bad seed digit
    shows up as a visible anomaly instead of a missing row.
    """
    thirteen = to_isbn13(raw_isbn)
    if thirteen:
        return f"isbn13:{thirteen}", "isbn13"
    digits = re.sub(r"[^0-9Xx]", "", str(raw_isbn or "")).upper()
    if digits:
        return f"isbn10:{digits}", "isbn10_fallback"
    return None, "unusable"


def normalize_url(url: Any) -> str | None:
    """URL with scheme, host and trailing slash normalised and query dropped.

    Dropping the query is load-bearing for provenance matching: the detail
    page's seller URL from JSON-LD carries no query while the card URLs carry
    ``?ref_=nav_sflk_srp``, so 0 of 253 match before this and 252 of 253 after.
    """
    text = clean_text(url)
    match = re.match(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*)://(?P<host>[^/?#]+)(?P<path>[^?#]*)", text)
    if not match:
        return None
    path = match.group("path").rstrip("/") or "/"
    return f"{match.group('scheme').lower()}://{match.group('host').lower()}{path}"


def seller_key(url: Any, name: Any = None, location: Any = None) -> tuple[str | None, str]:
    """``seller_id`` and how it was derived.

    The numeric storefront id is the key, not the name: one seller's slug embeds
    its location (``Book-Lovers-Warehouse-Watauga-TN-U.S.A`` vs
    ``Book-Lovers-Warehouse-Johnson-City-TN``) so the slug changes, while
    ``GreatBookPrices`` and ``GreatBookPricesUK`` are different sellers sharing
    a name prefix.  The id is stable under both.
    """
    normalized = normalize_url(url)
    if normalized:
        match = re.search(r"/(\d+)/sf$", normalized)
        if match:
            return f"abebooks:{match.group(1)}", "storefront_id"
    if clean_text(name):
        return "name_loc:" + stable_hash(normalize(name), normalize(location or "")), "name_location"
    if normalized:
        return "url:" + stable_hash(normalized), "url_only"
    return None, "missing"


def listing_key(listing_id: Any) -> str | None:
    text = clean_text(listing_id)
    return f"abebooks:{text}" if text else None


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #

def image_kind(image_url: Any) -> str | None:
    url = clean_text(image_url).lower()
    if not url:
        return None
    if SELLER_PHOTO_MARKER in url:
        return "seller_photo"
    if STOCK_COVER_MARKER in url:
        return "stock_cover"
    return "placeholder"


def has_real_image(listing: dict[str, Any]) -> bool:
    """True for a photo of the actual copy, not a publisher cover or placeholder."""
    return (image_kind(listing.get("image_url")) == "seller_photo"
            and listing.get("stock_image") is not True)


# --------------------------------------------------------------------------- #
# representative listing
# --------------------------------------------------------------------------- #

def match_detail_listing(book: dict[str, Any], listings: list[dict[str, Any]]) -> dict[str, Any]:
    """Which card the detail page describes, or None when it cannot be pinned.

    This is identification, not selection: the detail page exists and belongs to
    exactly one card, so the only question is which.  Every fallback is recorded
    so an unattributable record is visible rather than silently plausible.
    """
    target = normalize_url(book.get("seller_url"))
    if target:
        hits = [index for index, item in enumerate(listings)
                if normalize_url(item.get("seller_url")) == target]
        if hits:
            return {"indices": hits, "method": "seller_url"}

    name = normalize(book.get("seller_name"))
    if name:
        hits = [index for index, item in enumerate(listings)
                if normalize(item.get("seller_name")) == name]
        if hits:
            return {"indices": hits, "method": "seller_name"}

    title, authors = normalize(book.get("title")), normalize(book.get("authors"))
    if title:
        hits = [index for index, item in enumerate(listings)
                if normalize(item.get("listing_title")) == title
                and normalize(item.get("listing_authors")) == authors]
        if hits:
            return {"indices": hits, "method": "title_authors"}

    return {"indices": [], "method": "card_index_default"}


def _sort_key(index: int, listing: dict[str, Any], detail_index: int | None,
              prefer_seller_image: bool) -> tuple:
    """Card ranking, first differing key wins.

    ``seller_rating`` is deliberately absent: it is a coarse badge that is
    near-constant (5-star on 2288 of 2586 cards) and carries no information about
    whether the bibliographic fields are right -- exactly the inference the plan
    doc forbids.
    """
    is_detail = detail_index is not None and index == detail_index
    real_image = has_real_image(listing)
    price = listing.get("price")
    image_first, detail_second = (-int(real_image), -int(is_detail)) if prefer_seller_image \
        else (-int(is_detail), -int(real_image))
    return (
        image_first,
        detail_second,
        -int(bool(clean_text(listing.get("listing_authors")))),
        -int(price is not None),
        price if price is not None else float("inf"),
        index,
    )


def select_listing(book: dict[str, Any], search: dict[str, Any],
                   prefer_seller_image: bool = False) -> dict[str, Any]:
    """The representative listing for one book, with the reason it won."""
    listings = search.get("listings") or []
    match = match_detail_listing(book, listings)
    detail_index = min(match["indices"]) if match["indices"] else None
    detail_listing_id = (listing_key(listings[detail_index].get("listing_id"))
                         if detail_index is not None else None)

    if not listings:
        return {"listing": None, "index": None, "detail_index": None,
                "detail_listing_id": None, "detail_match_method": match["method"],
                "detail_listing_ambiguous": False, "detail_listing_candidate_count": 0,
                "selected_reason": "detail_unmatched"}

    index = min(range(len(listings)),
                key=lambda position: _sort_key(position, listings[position],
                                               detail_index, prefer_seller_image))
    winner = listings[index]
    any_real_image = any(has_real_image(item) for item in listings)
    if detail_index is None:
        reason = "detail_unmatched"
    elif index != detail_index:
        reason = "seller_image_preferred"
    elif has_real_image(winner):
        reason = "evidence_complete"
    elif any_real_image:
        reason = "seller_photo_elsewhere"
    else:
        reason = "stock_image_only"

    return {
        "listing": winner,
        "index": index,
        "detail_index": detail_index,
        "detail_listing_id": detail_listing_id,
        "detail_match_method": match["method"],
        "detail_listing_ambiguous": len(match["indices"]) > 1,
        "detail_listing_candidate_count": len(match["indices"]),
        "selected_reason": reason,
    }


# --------------------------------------------------------------------------- #
# value cleaning
# --------------------------------------------------------------------------- #

def optional_text(value: Any, max_chars: int) -> str | None:
    """Cleaned, length-capped text, or None -- never an empty string, so a
    missing value is distinguishable from an empty one."""
    return sanitize_cell_text(value, max_chars) or None


def sentinel_free_text(value: Any, max_chars: int) -> str | None:
    """As :func:`optional_text`, but a placeholder counts as absent."""
    text = optional_text(value, max_chars)
    return None if text and normalize(text) in _SENTINEL_KEYS else text


def as_year(value: Any) -> int | None:
    match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", clean_text(value))
    return int(match.group(1)) if match else None


def as_float(value: Any) -> float | None:
    """First number in the text, or None.  Never coerces a non-number to 0."""
    match = re.search(r"-?\d+(?:\.\d+)?", clean_text(value).replace(",", ""))
    return float(match.group(0)) if match else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return int(number) if number is not None else None


def iso_date(value: Any) -> str | None:
    """``"March 24, 2009"`` -> ``"2009-03-24"``, else None."""
    match = re.match(r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})",
                     clean_text(value))
    if not match or match.group("month") not in MONTHS:
        return None
    return f"{match.group('year')}-{MONTHS[match.group('month')]:02d}-{int(match.group('day')):02d}"


def result_count(text: Any) -> int | None:
    match = re.search(r"(\d[\d,]*)\s+result", clean_text(text), re.I)
    return int(match.group(1).replace(",", "")) if match else None


def edition_marker(title: Any) -> str | None:
    lowered = clean_text(title).lower()
    return next((marker for marker in EDITION_MARKERS if marker in lowered), None)


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #

def book_edition_row(record: dict[str, Any], selection: dict[str, Any],
                     book_id: str, scheme: str, max_chars: int) -> dict[str, Any]:
    """One normalised edition, from the detail page.

    ``title``/``authors`` carry their source because they are a seller's claim
    reproduced verbatim from the chosen card, not an edition truth.
    """
    book = record.get("book") or {}
    winner = selection["listing"] or {}
    image_url = book.get("catalogue_image_url")
    return {
        "book_id": book_id,
        "isbn_scheme": scheme,
        "isbn10": optional_text(book.get("isbn10"), max_chars),
        "isbn13": optional_text(book.get("isbn13"), max_chars),
        "title": optional_text(book.get("title"), max_chars),
        "authors": optional_text(book.get("authors"), max_chars),
        "title_source": "abebooks_detail_page",
        "authors_source": "abebooks_detail_page",
        "detail_listing_seller_id": seller_key(book.get("seller_url"), book.get("seller_name"))[0],
        "publisher": optional_text(book.get("publisher"), max_chars),
        "publication_year": as_year(book.get("publication_year")),
        "publication_year_raw": optional_text(book.get("publication_year"), max_chars),
        "language": optional_text(book.get("language"), max_chars),
        "binding": optional_text(book.get("binding"), max_chars),
        "edition_number": optional_text(book.get("edition_number"), max_chars),
        "series": optional_text(book.get("series"), max_chars),
        "dust_jacket": optional_text(book.get("dust_jacket"), max_chars),
        "product_type": None,  # SWS-only; kept so the column exists when it lands
        "dimensions": sentinel_free_text(book.get("dimensions"), max_chars),
        "item_weight": sentinel_free_text(book.get("item_weight"), max_chars),
        "copy_condition_grade": optional_text(book.get("condition"), max_chars),
        "condition_source": "abebooks_detail_page",
        "catalogue_image_url": optional_text(image_url, max_chars),
        "catalogue_image_kind": image_kind(image_url),
        "synopsis_text": optional_text(book.get("synopsis_text"), max_chars),
        "about_author_text": optional_text(book.get("about_author_text"), max_chars),
        "goodreads_rating": as_float(book.get("goodreads_rating")),
        "goodreads_rating_count": as_int(book.get("goodreads_rating_count")),
        "representative_listing_id": listing_key(winner.get("listing_id")),
        "source_url": record.get("book", {}).get("source_url"),
        "retrieved_at": record.get("retrieved_at"),
    }


def book_listing_row(record: dict[str, Any], selection: dict[str, Any], book_id: str,
                     seller_id: str | None, max_chars: int) -> dict[str, Any]:
    """The representative listing for one book.

    The detail-page-only columns are filled **only** when the winner is the card
    the detail page describes.  Attributing one seller's shipping price and
    inventory number to a different seller's card would be a fabricated join, so
    under ``--prefer-seller-image`` those columns go null instead.
    """
    book = record.get("book") or {}
    search = record.get("search") or {}
    winner = selection["listing"] or {}
    from_detail_page = selection["index"] is not None and selection["index"] == selection["detail_index"]

    price, currency = winner.get("price"), winner.get("currency")
    shipping_price = book.get("shipping_price") if from_detail_page else None
    shipping_currency = book.get("shipping_currency") if from_detail_page else None
    total_price = (price + shipping_price
                   if price is not None and shipping_price is not None
                   and currency and currency == shipping_currency else None)

    image_url = winner.get("image_url")
    return {
        "listing_id": listing_key(winner.get("listing_id")),
        "book_id": book_id,
        "seller_id": seller_id,
        "listing_url": winner.get("listing_url"),
        "seller_inventory_no": optional_text(book.get("seller_inventory_no"), max_chars) if from_detail_page else None,
        "listing_title": optional_text(winner.get("listing_title"), max_chars),
        "listing_authors": optional_text(winner.get("listing_authors"), max_chars),
        "condition": optional_text(winner.get("condition"), max_chars),
        "condition_description": None,  # no free-text condition field exists
        "availability_quantity": winner.get("availability_quantity"),
        "price": price,
        "currency": currency,
        "shipping_price": shipping_price,
        "shipping_currency": shipping_currency,
        "total_price": total_price,
        "vendor_description": optional_text(book.get("vendor_description"), max_chars) if from_detail_page else None,
        "vendor_image_url": optional_text(image_url, max_chars),
        "image_kind": image_kind(image_url),
        "stock_image_flag": winner.get("stock_image"),
        "edition_marker": edition_marker(winner.get("listing_title")),
        "selected_reason": selection["selected_reason"],
        "detail_listing_id": selection["detail_listing_id"],
        "detail_listing_ambiguous": selection["detail_listing_ambiguous"],
        "detail_listing_candidate_count": selection["detail_listing_candidate_count"],
        "detail_match_method": selection["detail_match_method"],
        "cards_truncated": _cards_truncated(search),
        "retrieved_at": record.get("retrieved_at"),
    }


def _cards_truncated(search: dict[str, Any]) -> bool:
    """True when the page reported more results than the captured first page."""
    reported = result_count(search.get("result_count_text"))
    return reported is not None and reported > (search.get("listing_count") or 0)


def seller_rows(records: list[dict[str, Any]], selections: dict[str, dict[str, Any]],
                max_chars: int) -> list[dict[str, Any]]:
    """One row per seller referenced by a representative listing.

    A seller is observed once per book they sell, so the observations are
    collapsed to one row: the richest observation wins, ties going to the later
    one.  Picking a single observation rather than merging fields keeps one
    timestamp describing every value in the row, instead of silently combining
    values seen at different times.

    Storefront pages were never fetched, so the seller's own columns come from
    the one detail page that introduced them; ``seller_page_fetched`` says so
    rather than letting ``seller_fields_retrieved_at`` imply a visit that never
    happened.  Under image-preferred anchoring the representative listing can
    belong to a different seller than the detail page, and that seller is then
    known by name, URL and badge alone.

    ``seller_name_variants`` is collected from every card in the file, which is
    how the surface-name variation the plan warns about stays visible without
    ever being used as a key.  Two ids sharing a normalised name are flagged,
    never merged: they can be genuinely different sellers (GreatBookPrices vs
    GreatBookPricesUK).
    """
    variants: dict[str, set[str]] = {}
    for record in records:
        search = record.get("search") or {}
        book = record.get("book") or {}
        for item in list(search.get("listings") or []) + [book]:
            key = seller_key(item.get("seller_url"), item.get("seller_name"))[0]
            name = optional_text(item.get("seller_name"), max_chars)
            if key and name:
                variants.setdefault(key, set()).add(name)

    by_name: dict[str, set[str]] = {}
    for key, names in variants.items():
        for name in names:
            by_name.setdefault(normalize(name), set()).add(key)

    observations: dict[str, list[tuple[str, int, dict[str, Any]]]] = {}
    for order, record in enumerate(records):
        selection = selections.get(record.get("isbn"))
        if not selection or selection["listing"] is None:
            continue
        book = record.get("book") or {}
        listing = selection["listing"]
        seller_id, method = seller_key(listing.get("seller_url"), listing.get("seller_name"))
        if not seller_id:
            continue
        # The seller's own columns live on the detail page, and the detail page
        # belongs to exactly one seller.
        from_detail = seller_id == seller_key(book.get("seller_url"), book.get("seller_name"))[0]
        source = book if from_detail else listing
        names = sorted(variants.get(seller_id, set()))
        row = {
            "seller_id": seller_id,
            "seller_name": optional_text(source.get("seller_name"), max_chars),
            "seller_name_variants": names or None,
            "seller_name_collision": any(len(by_name.get(normalize(n), set())) > 1 for n in names),
            "seller_key_method": method,
            "seller_url": normalize_url(listing.get("seller_url") or book.get("seller_url")),
            "location_raw": optional_text(book.get("seller_location"), max_chars) if from_detail else None,
            # All three are null together when the address line has fewer than
            # three comma parts ("Lincoln, United Kingdom").  That is the page's
            # granularity, so it is preserved rather than patched.
            "city": optional_text(book.get("seller_city"), max_chars) if from_detail else None,
            "region": optional_text(book.get("seller_region"), max_chars) if from_detail else None,
            "country": optional_text(book.get("seller_country"), max_chars) if from_detail else None,
            "seller_rating": optional_text(source.get("seller_rating"), max_chars),
            "seller_rating_kind": "abebooks_badge_1_5",
            "seller_since": optional_text(book.get("seller_since"), max_chars) if from_detail else None,
            "seller_since_iso": iso_date(book.get("seller_since")) if from_detail else None,
            "specialties": None,  # SWS-only
            "terms_of_sale": optional_text(book.get("seller_terms"), max_chars) if from_detail else None,
            "shipping_terms": optional_text(book.get("shipping_terms"), max_chars) if from_detail else None,
            "seller_description": optional_text(book.get("seller_description"), max_chars) if from_detail else None,
            "fields_from_detail_page": from_detail,
            "seller_fields_retrieved_at": record.get("retrieved_at"),
            "seller_page_fetched": False,
        }
        observations.setdefault(seller_id, []).append((record.get("retrieved_at") or "", order, row))

    rows: list[dict[str, Any]] = []
    for seller_id, group in observations.items():
        best = max(group, key=lambda item: (_richness(item[2]), item[0], item[1]))[2]
        rows.append({**best, "observations": len(group)})
    rows.sort(key=lambda row: row["seller_id"])
    return rows


def _richness(row: dict[str, Any]) -> int:
    return sum(1 for value in row.values() if value is not None)


def evidence_rows(record: dict[str, Any], selection: dict[str, Any], book_id: str | None,
                  max_chars: int) -> list[dict[str, Any]]:
    """Every asset that backs a row, with the page it came from.

    On text rows ``sha256``/``local_path`` are the **containing page's**, not a
    hash of the excerpt -- the excerpt is a projection of that page and hashing
    it would make the column mean something else than it does everywhere else.

    Page snapshots are emitted for every record with an HTML snapshot, including
    ones no table row came from: a degraded page is worthless alone and
    indispensable next to the same URL fetched successfully later, and a
    zero-result search page is the evidence that AbeBooks had no copies.  Those
    rows carry ``record_isbn`` and a null ``book_id``.
    """
    rows: list[dict[str, Any]] = []
    isbn = record.get("isbn")
    retrieved_at = record.get("retrieved_at")
    book = record.get("book") or {}
    search = record.get("search") or {}

    def add(asset_type: str, uri: Any, local_path: Any, sha256: Any, mime: Any,
            source_page: Any, locator: Any, text: Any = None,
            method: str | None = None, confidence: Any = None,
            listing_id: str | None = None, representative: bool | None = None,
            fetch_status: str = "page_saved") -> None:
        params = (book_id, asset_type, uri, local_path, listing_id, len(rows))
        rows.append({
            "evidence_id": "ev:" + stable_hash(*params),
            "book_id": book_id,
            "record_isbn": isbn,
            "listing_id": listing_id,
            "is_representative_listing": representative,
            "asset_type": asset_type,
            "uri": clean_text(uri) or None,
            "local_path": clean_text(local_path) or None,
            "sha256": clean_text(sha256) or None,
            "mime_type": mime,
            "source_page": clean_text(source_page) or None,
            "source_locator": locator,
            "caption_or_alt": None,
            "extracted_text": optional_text(text, max_chars),
            "text_chars": len(clean_text(text)) or None,
            "extraction_method": method,
            "confidence": confidence,
            "fetch_status": fetch_status,
            "retrieved_at": retrieved_at,
        })

    if record.get("html_path"):
        add("search_page", search.get("source_url"), record.get("html_path"),
            record.get("html_sha256"), "text/html", search.get("source_url"),
            "page snapshot", method="http_snapshot", confidence=1.0)
    if record.get("book_html_path"):
        add("detail_page", book.get("source_url"), record.get("book_html_path"),
            record.get("book_html_sha256"), "text/html", book.get("source_url"),
            "page snapshot", method="http_snapshot", confidence=1.0)
    if record.get("degraded_html_path"):
        add("degraded_page", record.get("degraded_url"), record.get("degraded_html_path"),
            record.get("degraded_html_sha256"), "text/html", record.get("degraded_url"),
            "page snapshot", method="http_snapshot", confidence=1.0)

    if book:
        winner = selection["listing"] or {}
        representative_id = listing_key(winner.get("listing_id"))
        for asset_type, key, locator in TEXT_ASSETS:
            text = book.get(key)
            if clean_text(text):
                add(asset_type, book.get("source_url"), record.get("book_html_path"),
                    record.get("book_html_sha256"), "text/html", book.get("source_url"),
                    locator, text=text, method="rule_selector", confidence=1.0,
                    listing_id=representative_id, representative=True)

        cover = book.get("catalogue_image_url")
        if clean_text(cover):
            # The edition's cover is sometimes the detail listing's own photo, in
            # which case it is a seller photo and belongs to the listing.
            kind = image_kind(cover)
            add("catalogue_cover" if kind == "stock_cover" else "seller_cover",
                cover, None, None, None, book.get("source_url"), kind,
                listing_id=representative_id if kind == "seller_photo" else None,
                representative=kind == "seller_photo",
                fetch_status="url_only")

        # Real photos owned by cards other than the representative one.  Emitted
        # because the book's only actual photo is often there, and the
        # representative flag states plainly that the listing_id is not the
        # representative row.
        seen = {clean_text(cover)}
        for item in search.get("listings") or []:
            url = item.get("image_url")
            if not has_real_image(item) or clean_text(url) in seen:
                continue
            seen.add(clean_text(url))
            listing = listing_key(item.get("listing_id"))
            add("seller_cover", url, None, None, None, search.get("source_url"),
                "seller_photo", listing_id=listing,
                representative=listing == representative_id, fetch_status="url_only")

    return rows


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def build_tables(records: list[dict[str, Any]], seed_isbns: list[str] | None = None,
                 prefer_seller_image: bool = False,
                 max_chars: int = 1024) -> dict[str, Any]:
    """The four tables plus stats, from already-loaded records."""
    winners, superseded = dedupe_records(records)

    editions: list[dict[str, Any]] = []
    listings: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    no_listings: list[str] = []
    anomalies: list[dict[str, Any]] = []
    seller_counts: dict[str, int] = {}

    for record in winners:
        isbn = record.get("isbn") or ""
        status = record.get("status")
        book_id, scheme = book_id_for(isbn)
        book = record.get("book") or {}
        search = record.get("search") or {}

        if status in ("degraded", "blocked", "error"):
            # Unknown, not absent: only a real zero-result page is a fact.
            unresolved.append({
                "isbn": isbn, "book_id": book_id, "status": status,
                "http_status": record.get("http_status"),
                "error": record.get("error"),
                "attempts": sum(1 for r in records if r.get("isbn") == isbn),
                "source_url": record.get("source_url"),
                "retrieved_at": record.get("retrieved_at"),
            })
            # Its page snapshots still go in the evidence table: an unhydrated
            # page is worthless alone and indispensable next to the same URL
            # fetched successfully later.  No book row exists, so book_id is
            # null and record_isbn carries the identity.
            evidence.extend(evidence_rows(record, {"listing": None}, None, max_chars))
            continue

        if status == "no_listings":
            no_listings.append(isbn)
            evidence.extend(evidence_rows(record, {"listing": None}, None, max_chars))
            continue

        if status != "ok":
            anomalies.append({"isbn": isbn, "issue": f"unhandled status {status!r}"})
            continue

        if book.get("isbn13") and to_isbn13(book["isbn13"]) != to_isbn13(isbn):
            # Free consistency check: the seed and the page agree 253/253 today,
            # so any disagreement is a real signal, not noise.
            anomalies.append({
                "isbn": isbn,
                "issue": "page isbn13 does not match the seed ISBN",
                "page_isbn13": book.get("isbn13"),
            })

        selection = select_listing(book, search, prefer_seller_image)
        selections[isbn] = selection
        editions.append(book_edition_row(record, selection, book_id, scheme, max_chars))
        if selection["listing"] is None:
            anomalies.append({"isbn": isbn, "issue": "ok record has no listings"})
            evidence.extend(evidence_rows(record, selection, book_id, max_chars))
            continue
        seller_id = seller_key(book.get("seller_url"), book.get("seller_name"))[0]
        listings.append(book_listing_row(record, selection, book_id, seller_id, max_chars))
        evidence.extend(evidence_rows(record, selection, book_id, max_chars))

    # Report what the other anchoring would have covered, so the cost of the
    # chosen rule is a number in the output rather than an argument.
    for label, prefer in (("detail_anchored", False), ("seller_image_preferred", True)):
        keys = set()
        for record in winners:
            if record.get("status") != "ok" or not record.get("book"):
                continue
            other = select_listing(record["book"], record.get("search") or {}, prefer)
            if other["listing"] is not None:
                keys.add(seller_key(other["listing"].get("seller_url"),
                                    other["listing"].get("seller_name"))[0])
        seller_counts[label] = len(keys)

    sellers = seller_rows([r for r in winners if r.get("status") == "ok"], selections, max_chars)
    artifact_tables = {
        "book_edition": editions,
        "seller": sellers,
        "book_listing": listings,
        "evidence_asset": evidence,
    }

    stats: dict[str, Any] = {
        "counts": {name: len(rows) for name, rows in artifact_tables.items()},
        "status_counts": _status_counts(records),
        "distinct_isbns": len({r.get("isbn") for r in records}),
        "no_listings_isbns": no_listings,
        "unresolved_isbns": [row["isbn"] for row in unresolved],
        "superseded_records": len(superseded),
        "selected_reasons": _tally(row["selected_reason"] for row in listings),
        "detail_match_methods": _tally(row["detail_match_method"] for row in listings),
        "image_kinds": _tally(row["image_kind"] for row in listings),
        "ambiguous_detail_listing": sum(1 for row in listings if row["detail_listing_ambiguous"]),
        "cards_truncated": sum(1 for row in listings if row["cards_truncated"]),
        "seller_count_by_anchoring": seller_counts,
        "evidence_asset_types": _tally(row["asset_type"] for row in evidence),
        "anomalies": anomalies,
        "field_coverage": {name: _coverage(rows) for name, rows in artifact_tables.items()},
    }
    if seed_isbns:
        seed = set(seed_isbns)
        represented = {row["isbn"] for row in winners}
        stats["seed"] = {
            "isbns": len(seed),
            "queried": len(seed & represented),
            "coverage": round(len(seed & represented) / len(seed), 4) if seed else None,
            "records_not_in_seed": sorted(represented - seed),
        }
    return {**artifact_tables, "unresolved": unresolved, "superseded": superseded, "stats": stats}


def _status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return _tally(record.get("status") for record in records)


def _tally(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value if value is not None else "null"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Share of rows with a non-null value, per column."""
    if not rows:
        return {}
    columns = list(rows[0])
    return {
        column: round(sum(1 for row in rows if row.get(column) is not None) / len(rows), 4)
        for column in columns
    }


def split_book_ids(book_ids: list[str], seed: int = 13,
                   ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, Any]:
    """Deterministic train/dev/test assignment keyed on ``book_id``.

    Keyed on the book, not the row, so a book's evidence can never straddle two
    splits.  Hashing rather than shuffling keeps the assignment stable when the
    collection grows: a new book joins without moving existing ones.
    """
    train_ratio, dev_ratio = ratios[0], ratios[1]
    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for book_id in sorted(book_ids):
        position = int(stable_hash(book_id, seed, length=8), 16) / 0xFFFFFFFF
        split = "train" if position < train_ratio else "dev" if position < train_ratio + dev_ratio else "test"
        splits[split].append(book_id)
    return {"split_key": "book_id", "ratios": list(ratios), "seed": seed, **splits}

"""Tests for the four-table AbeBooks build.

Fixtures are minimal synthetic records -- the point is the rules (which card
wins, which key survives, what a missing value becomes), not page parsing, which
`tests/test_abebooks_scraper.py` already covers.
"""

import json
from pathlib import Path

import pytest

from abebooks_tables import (
    book_id_for,
    build_tables,
    dedupe_records,
    image_kind,
    normalize_url,
    select_listing,
    seller_key,
    sentinel_free_text,
    split_book_ids,
    to_isbn13,
)

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

SHELF_PHOTO = "https://pictures.abebooks.com/inventory/32510370781.jpg"
STOCK_COVER = "https://pictures.abebooks.com/isbn/9780201853940-us._SL300_.jpg"
DETAIL_SELLER = "https://www.abebooks.com/Greener-Books-London/52838368/sf"
OTHER_SELLER = "https://www.abebooks.com/Book-Lovers-Warehouse/3207104/sf"


def card(listing_id, *, seller_url=DETAIL_SELLER, seller_name="Greener Books",
         price=19.0, image=STOCK_COVER, stock=True, authors="Knuth, Donald E.",
         title="Art of Computer Programming", condition="Used - Very good"):
    return {
        "listing_id": listing_id,
        "listing_url": f"https://www.abebooks.com/x/{listing_id}/bd",
        "listing_title": title,
        "listing_authors": authors,
        "seller_name": seller_name,
        # the query string is what the real cards carry, and dropping it is what
        # makes the detail page's JSON-LD seller URL match
        "seller_url": f"{seller_url}?ref_=nav_sflk_srp",
        "condition": condition,
        "price": price,
        "currency": "USD",
        "availability_quantity": 1,
        "seller_rating": "5-star seller",
        "image_url": image,
        "stock_image": stock,
    }


def book(**overrides):
    base = {
        "source_url": "https://www.abebooks.com/x/1/bd",
        "title": "Art of Computer Programming",
        "authors": "Knuth, Donald E.",
        "isbn10": "0201853949",
        "isbn13": "9780201853940",
        "publisher": "Addison Wesley",
        "publication_year": "2005",
        "language": "English",
        "binding": "Paperback",
        "edition_number": "First Edition.",
        "series": None,
        "dust_jacket": "Yes",
        "dimensions": "N/A",
        "item_weight": "1,400 grams",
        "condition": "Good",
        "catalogue_image_url": STOCK_COVER,
        "stock_image": True,
        "synopsis_text": "A treatise.",
        "about_author_text": "Donald Knuth is a computer scientist.",
        "vendor_description": "The item might be beaten up but readable.",
        "goodreads_rating": "4.54",
        "goodreads_rating_count": "46",
        "seller_name": "Greener Books",
        "seller_url": DETAIL_SELLER,
        "seller_since": "October 31, 2007",
        "seller_description": "An enterprise.",
        "seller_location": "London, LND, United Kingdom",
        "seller_city": "London",
        "seller_region": "LND",
        "seller_country": "United Kingdom",
        "seller_rating": "5-star seller",
        "seller_terms": "Returns accepted within 30 days.",
        "shipping_terms": "Ships in 4-14 business days.",
        "shipping_price": 6.0,
        "shipping_currency": "USD",
        "seller_inventory_no": "RWARE0000058606",
    }
    base.update(overrides)
    return base


def record(isbn="0201853949", *, status="ok", retrieved_at="2026-09-13T10:00:00+00:00",
           cards=None, detail=None, **record_fields):
    """A scrape record shaped like the ones the collector writes."""
    cards = [card(1)] if cards is None else cards
    result = {
        "isbn": isbn,
        "retrieved_at": retrieved_at,
        "status": status,
        "source_url": f"https://www.abebooks.com/servlet/SearchResults?isbn={isbn}",
        "http_status": 200,
    }
    if status in ("ok", "no_listings"):
        result["search"] = {
            "source_url": result["source_url"],
            "result_count_text": f"({len(cards)} results)",
            "listing_count": len(cards),
            "listings": cards,
        }
        result["html_path"] = f"html/{isbn}.html"
        result["html_sha256"] = "a" * 64
    if status == "ok":
        result["book"] = book() if detail is None else detail
        result["book_html_path"] = f"html/{isbn}_book.html"
        result["book_html_sha256"] = "b" * 64
    result.update(record_fields)
    return result


def write_jsonl(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# ISBN keys
# --------------------------------------------------------------------------- #

def test_to_isbn13_converts_a_valid_isbn10():
    assert to_isbn13("0201853949") == "9780201853940"


def test_to_isbn13_keeps_a_valid_isbn13():
    assert to_isbn13("9780201853940") == "9780201853940"


@pytest.mark.parametrize("bad", ["0201853948", "9780201853941", "123", "", None])
def test_to_isbn13_rejects_a_bad_checksum(bad):
    assert to_isbn13(bad) is None


def test_book_id_keys_on_isbn13_and_flags_an_invalid_seed():
    assert book_id_for("0201853949") == ("isbn13:9780201853940", "isbn13")
    # a seed with a bad check digit is kept visibly rather than dropped
    assert book_id_for("0201853948") == ("isbn10:0201853948", "isbn10_fallback")


# --------------------------------------------------------------------------- #
# seller keys
# --------------------------------------------------------------------------- #

def test_normalize_url_drops_the_query_that_breaks_provenance_matching():
    # without this, 0 of 253 detail-page seller URLs match their card
    assert normalize_url(f"{DETAIL_SELLER}?ref_=nav_sflk_srp") == DETAIL_SELLER
    assert normalize_url(DETAIL_SELLER + "/") == DETAIL_SELLER


def test_seller_key_uses_the_numeric_id_not_the_slug():
    key, method = seller_key("https://www.abebooks.com/Book-Lovers-Warehouse-Johnson-City-TN/3207104/sf")
    assert key == "abebooks:3207104"
    assert method == "storefront_id"
    # the same seller under a location-bearing slug is the same seller
    assert seller_key("https://www.abebooks.com/Other-Slug-Name/3207104/sf")[0] == key


def test_seller_key_falls_back_when_there_is_no_storefront_id():
    assert seller_key(None, "Greener Books", "London")[1] == "name_location"
    assert seller_key("https://www.abebooks.com/x/y")[1] == "url_only"
    assert seller_key(None, None)[0] is None


# --------------------------------------------------------------------------- #
# image kind
# --------------------------------------------------------------------------- #

def test_image_kind_reads_the_url_not_the_flag():
    # 13 real cards say stock_image=False while serving the publisher cover
    assert image_kind(STOCK_COVER) == "stock_cover"
    assert image_kind(SHELF_PHOTO) == "seller_photo"
    assert image_kind("https://assets.prod.abebookscdn.com/x.jpg") == "placeholder"
    assert image_kind(None) is None


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

def test_select_listing_anchors_on_the_detail_listing_even_when_a_better_image_exists():
    cards = [
        card(1, image=STOCK_COVER, stock=True),
        card(2, seller_url=OTHER_SELLER, seller_name="Book Lover's Warehouse",
             image=SHELF_PHOTO, stock=False, authors="Donald E. Knuth"),
    ]
    selection = select_listing(book(), {"listings": cards}, prefer_seller_image=False)
    assert selection["index"] == 0
    assert selection["detail_listing_id"] == "abebooks:1"
    assert selection["detail_match_method"] == "seller_url"
    assert selection["selected_reason"] == "seller_photo_elsewhere"


def test_prefer_seller_image_swaps_the_anchor_and_says_so():
    cards = [
        card(1, image=STOCK_COVER, stock=True),
        card(2, seller_url=OTHER_SELLER, seller_name="Book Lover's Warehouse",
             image=SHELF_PHOTO, stock=False),
    ]
    selection = select_listing(book(), {"listings": cards}, prefer_seller_image=True)
    assert selection["index"] == 1
    assert selection["selected_reason"] == "seller_image_preferred"
    # the detail listing is still identified, so the audit trail survives the swap
    assert selection["detail_listing_id"] == "abebooks:1"


def test_select_listing_reports_an_ambiguous_detail_match():
    # two cards for one seller: the detail page cannot be pinned to either
    cards = [card(1), card(2, price=21.0)]
    selection = select_listing(book(), {"listings": cards})
    assert selection["detail_listing_ambiguous"] is True
    assert selection["detail_listing_candidate_count"] == 2
    assert selection["detail_listing_id"] == "abebooks:1"


def test_select_listing_matches_a_renamed_seller_by_name():
    detail = book(seller_url="https://www.abebooks.com/Greener-Books/999/sf")
    cards = [card(1), card(2, seller_url=OTHER_SELLER, seller_name="Other Books")]
    selection = select_listing(detail, {"listings": cards})
    assert selection["detail_match_method"] == "seller_name"
    assert selection["detail_listing_id"] == "abebooks:1"


def test_select_listing_falls_back_to_the_default_when_nothing_matches():
    detail = book(seller_url="https://www.abebooks.com/Nobody/1/sf", seller_name="Nobody")
    # titles and authors differ too, so no fallback can identify the card
    cards = [card(1, title="Some Other Book", authors="Someone Else"),
             card(2, seller_url=OTHER_SELLER, seller_name="Other Books",
                  title="Some Other Book", authors="Someone Else")]
    selection = select_listing(detail, {"listings": cards})
    assert selection["detail_index"] is None
    assert selection["detail_match_method"] == "card_index_default"
    assert selection["selected_reason"] == "detail_unmatched"


def test_select_listing_prefers_a_real_photo_over_a_placeholder_for_a_tie():
    # neither card is the detail listing, so the image is the first real signal
    detail = book(seller_url="https://www.abebooks.com/Nobody/1/sf", seller_name="Nobody")
    cards = [
        card(1, title="Some Other Book", authors="Someone Else",
             image="https://assets.prod.abebookscdn.com/none.jpg", stock=False),
        card(2, title="Some Other Book", authors="Someone Else",
             image=SHELF_PHOTO, stock=False),
    ]
    assert select_listing(detail, {"listings": cards})["index"] == 1


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #

def test_dedupe_prefers_ok_over_a_later_no_listings():
    records = [
        record(status="ok"),
        record(status="no_listings", retrieved_at="2026-09-13T11:00:00+00:00"),
    ]
    winners, superseded = dedupe_records(records)
    assert [r["status"] for r in winners] == ["ok"]
    assert len(superseded) == 1


def test_dedupe_takes_the_latest_record_at_the_same_status():
    records = [
        record(retrieved_at="2026-09-13T10:00:00+00:00", detail=book(publisher="Old")),
        record(retrieved_at="2026-09-13T12:00:00+00:00", detail=book(publisher="New")),
    ]
    winners, _ = dedupe_records(records)
    assert winners[0]["book"]["publisher"] == "New"


def test_dedupe_rejects_a_no_listings_verdict_the_page_does_not_support():
    # a throttled search page also parses to zero cards; accepting it would drop
    # a book we already had
    ok = record(status="ok")
    throttled = record(status="no_listings", retrieved_at="2026-09-13T11:00:00+00:00")
    throttled["search"]["result_count_text"] = None
    throttled["html_path"] = None
    winners, _ = dedupe_records([ok, throttled])
    assert winners[0]["status"] == "ok"


def test_dedupe_is_stable_when_the_file_grows():
    before, _ = dedupe_records([record(status="degraded")])
    after, _ = dedupe_records([record(status="degraded"), record(status="ok")])
    assert before[0]["status"] == "degraded"
    assert after[0]["status"] == "ok"


# --------------------------------------------------------------------------- #
# value cleaning
# --------------------------------------------------------------------------- #

def test_sentinel_free_text_drops_the_placeholder_that_inflates_coverage():
    # 36 of 46 non-null dimensions values are the literal string "N/A"
    assert sentinel_free_text("N/A", 1024) is None
    assert sentinel_free_text("Not Available", 1024) is None
    assert sentinel_free_text("22 cm", 1024) == "22 cm"
    assert sentinel_free_text(None, 1024) is None


def test_unparseable_numbers_become_null_never_zero():
    rows = build_tables([record(detail=book(publication_year="First Edition.",
                                             goodreads_rating="no rating"))])
    edition = rows["book_edition"][0]
    assert edition["publication_year"] is None
    assert edition["publication_year_raw"] == "First Edition."
    assert edition["goodreads_rating"] is None


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def test_build_tables_populates_the_detail_only_columns_from_the_detail_page():
    rows = build_tables([record()])
    listing = rows["book_listing"][0]
    assert listing["selected_reason"] == "stock_image_only"
    assert listing["shipping_price"] == 6.0
    assert listing["total_price"] == 25.0
    assert listing["seller_inventory_no"] == "RWARE0000058606"
    assert listing["vendor_description"] == "The item might be beaten up but readable."
    assert listing["book_id"] == "isbn13:9780201853940"
    assert listing["seller_id"] == "abebooks:52838368"


def test_build_tables_nulls_the_detail_only_columns_when_the_anchor_moves():
    # attributing one seller's shipping price to another seller's card would be
    # a fabricated join
    cards = [
        card(1, image=STOCK_COVER, stock=True),
        card(2, seller_url=OTHER_SELLER, seller_name="Book Lover's Warehouse",
             image=SHELF_PHOTO, stock=False),
    ]
    rows = build_tables([record(cards=cards)], prefer_seller_image=True)
    listing = rows["book_listing"][0]
    assert listing["listing_id"] == "abebooks:2"
    assert listing["shipping_price"] is None
    assert listing["total_price"] is None
    assert listing["vendor_description"] is None
    # card-level data is still the winning card's
    assert listing["price"] == 19.0


def test_build_tables_keeps_the_edition_from_the_page_not_the_seed():
    rows = build_tables([record()], seed_isbns=["0201853949"])
    edition = rows["book_edition"][0]
    assert edition["title"] == "Art of Computer Programming"
    assert edition["title_source"] == "abebooks_detail_page"
    assert edition["copy_condition_grade"] == "Good"
    assert edition["dimensions"] is None  # "N/A" is not data
    assert edition["item_weight"] == "1,400 grams"
    assert "page_count" not in edition
    assert "back_cover_text" not in edition
    assert rows["stats"]["seed"]["coverage"] == 1.0


def test_build_tables_excludes_no_listings_from_the_tables_but_records_it():
    rows = build_tables([record(isbn="0201853949"), record(isbn="0321335708", status="no_listings")])
    assert [r["book_id"] for r in rows["book_edition"]] == ["isbn13:9780201853940"]
    assert rows["stats"]["counts"]["book_listing"] == 1
    assert rows["stats"]["no_listings_isbns"] == ["0321335708"]


def test_build_tables_routes_unsettled_records_to_unresolved():
    rows = build_tables([record(isbn="0201616475", status="degraded")])
    assert rows["book_edition"] == []
    assert [r["status"] for r in rows["unresolved"]] == ["degraded"]
    assert rows["unresolved"][0]["book_id"] == "isbn13:9780201616477"


def test_build_tables_flags_a_page_isbn_that_disagrees_with_the_seed():
    rows = build_tables([record(detail=book(isbn13="9780000000000"))])
    assert any("does not match" in a["issue"] for a in rows["stats"]["anomalies"])


def test_seller_rows_report_the_key_method_and_the_never_fetched_page():
    rows = build_tables([record()])
    seller = rows["seller"][0]
    assert seller["seller_id"] == "abebooks:52838368"
    assert seller["seller_key_method"] == "storefront_id"
    assert seller["seller_page_fetched"] is False
    assert seller["terms_of_sale"] == "Returns accepted within 30 days."
    assert seller["seller_since_iso"] == "2007-10-31"
    assert seller["city"] == "London"
    assert seller["specialties"] is None


def test_seller_rows_collect_name_variants_from_every_card():
    cards = [card(1, seller_name="Greener Books"),
             card(2, seller_name="Greener Books London")]
    rows = build_tables([record(cards=cards)])
    assert rows["seller"][0]["seller_name_variants"] == ["Greener Books", "Greener Books London"]


def test_seller_rows_hold_one_row_per_seller_not_one_per_book():
    # the same seller carries four of the books, so four observations collapse
    records = [
        record(isbn=f"020185394{index}", detail=book(isbn13=f"978020185394{index}"))
        for index in range(4)
    ]
    rows = build_tables(records)
    assert len(rows["book_edition"]) == 4
    assert len(rows["seller"]) == 1
    assert rows["seller"][0]["observations"] == 4


def test_seller_rows_keep_the_richest_observation():
    # two different books from one seller: the later observation is the poorer
    # one, and must not overwrite the fuller fields seen earlier
    sparse = book(isbn10="0201616475", isbn13="9780201616477", seller_terms=None,
                  seller_city=None, seller_region=None, seller_country=None,
                  seller_location=None, seller_description=None)
    records = [
        record(isbn="0201616475", retrieved_at="2026-09-13T12:00:00+00:00", detail=sparse),
        record(isbn="0201853949", retrieved_at="2026-09-13T10:00:00+00:00"),
    ]
    seller = build_tables(records)["seller"][0]
    assert seller["observations"] == 2
    assert seller["terms_of_sale"] == "Returns accepted within 30 days."
    assert seller["seller_fields_retrieved_at"] == "2026-09-13T10:00:00+00:00"


def test_seller_rows_carry_card_only_fields_when_the_anchor_moves_off_the_detail_page():
    cards = [
        card(1, image=STOCK_COVER, stock=True),
        card(2, seller_url=OTHER_SELLER, seller_name="Book Lover's Warehouse",
             image=SHELF_PHOTO, stock=False),
    ]
    rows = build_tables([record(cards=cards)], prefer_seller_image=True)
    # one row per seller the representative listing references: the detail
    # page's seller sells no book here, so it is not a seller of this dataset
    by_id = {row["seller_id"]: row for row in rows["seller"]}
    assert set(by_id) == {"abebooks:3207104"}
    other = by_id["abebooks:3207104"]
    # the detail page never described this seller, so nothing is invented for it
    assert other["fields_from_detail_page"] is False
    assert other["seller_name"] == "Book Lover's Warehouse"
    assert other["city"] is None
    assert other["terms_of_sale"] is None
    # and the anchoring comparison reports the seller set each rule reaches --
    # one here, but a different one per rule
    assert rows["stats"]["seller_count_by_anchoring"] == {
        "detail_anchored": 1, "seller_image_preferred": 1}


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #

def test_evidence_pages_carry_the_snapshot_hash_and_text_carries_the_pages():
    rows = build_tables([record()])
    by_type = {}
    for row in rows["evidence_asset"]:
        by_type.setdefault(row["asset_type"], []).append(row)

    assert by_type["search_page"][0]["sha256"] == "a" * 64
    assert by_type["search_page"][0]["local_path"] == "html/0201853949.html"
    detail = by_type["detail_page"][0]
    assert detail["sha256"] == "b" * 64

    # a text asset's hash is the containing page's, not the excerpt's
    synopsis = by_type["synopsis"][0]
    assert synopsis["sha256"] == "b" * 64
    assert synopsis["source_locator"] == "ld+json Product.description"
    assert synopsis["extracted_text"] == "A treatise."
    assert synopsis["extraction_method"] == "rule_selector"


def test_evidence_keeps_the_degraded_snapshot_for_a_book_no_table_row_uses():
    rows = build_tables([record(isbn="0201616475", status="degraded",
                                degraded_url="https://www.abebooks.com/x/1/bd",
                                degraded_html_path="html/0201616475_degraded.html",
                                degraded_html_sha256="c" * 64)])
    page = rows["evidence_asset"][0]
    assert page["asset_type"] == "degraded_page"
    assert page["book_id"] is None
    assert page["record_isbn"] == "0201616475"


def test_evidence_image_rows_are_url_only_until_downloaded():
    cards = [
        card(1, image=STOCK_COVER, stock=True),
        card(2, seller_url=OTHER_SELLER, seller_name="Book Lover's Warehouse",
             image=SHELF_PHOTO, stock=False),
    ]
    rows = build_tables([record(cards=cards)])
    covers = [r for r in rows["evidence_asset"] if r["asset_type"] == "seller_cover"]
    assert [r["uri"] for r in covers] == [SHELF_PHOTO]
    # the book's only real photo belongs to a card that is not the representative
    assert covers[0]["listing_id"] == "abebooks:2"
    assert covers[0]["is_representative_listing"] is False
    assert covers[0]["fetch_status"] == "url_only"
    assert covers[0]["sha256"] is None
    assert covers[0]["local_path"] is None


def test_evidence_ids_are_unique_and_stable():
    first = build_tables([record()])["evidence_asset"]
    second = build_tables([record()])["evidence_asset"]
    assert [r["evidence_id"] for r in first] == [r["evidence_id"] for r in second]
    assert len({r["evidence_id"] for r in first}) == len(first)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #

def test_split_is_deterministic_and_never_splits_a_book():
    ids = [f"isbn13:97802018539{index:02d}" for index in range(40)]
    first = split_book_ids(ids, seed=13)
    assert first == split_book_ids(list(reversed(ids)), seed=13)
    placed = [book for split in ("train", "dev", "test") for book in first[split]]
    assert sorted(placed) == sorted(ids)
    assert len(set(placed)) == len(ids)

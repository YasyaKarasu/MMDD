import json
import random
import time
from pathlib import Path

import pytest

from src.abebooks_scraper import (
    Pace,
    book_is_empty,
    collect,
    detect_block,
    fetched_isbns,
    parse_book_page,
    parse_price,
    parse_search_page,
    read_isbns,
    resolve_proxy,
)

# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #

def test_read_isbns_deduplicates_and_strips(tmp_path: Path):
    p = tmp_path / "books.txt"
    p.write_text("s\t978-0-201-85394-9\tT\tA\ns2\t9780201853949\tT2\tA2\n", encoding="utf-8")
    assert read_isbns(p) == ["9780201853949"]


def test_read_isbns_honours_limit(tmp_path: Path):
    p = tmp_path / "books.txt"
    p.write_text("".join(f"s\t020185394{i}\tT\tA\n" for i in range(5)), encoding="utf-8")
    assert len(read_isbns(p, limit=3)) == 3


def test_read_isbns_skips_lines_without_isbn_column(tmp_path: Path):
    p = tmp_path / "books.txt"
    p.write_text("only-one-column\ns\t\tT\tA\ns\t0201853949\tT\tA\n", encoding="utf-8")
    assert read_isbns(p) == ["0201853949"]


# --------------------------------------------------------------------------- #
# proxy
# --------------------------------------------------------------------------- #

def test_resolve_proxy_prefers_explicit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://env:1")
    assert resolve_proxy("http://explicit:8080") == {"server": "http://explicit:8080"}


def test_resolve_proxy_normalises_socks5h(monkeypatch: pytest.MonkeyPatch):
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("all_proxy", "socks5h://127.0.0.1:7891")
    assert resolve_proxy(None) == {"server": "socks5://127.0.0.1:7891"}


def test_resolve_proxy_absent(monkeypatch: pytest.MonkeyPatch):
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    assert resolve_proxy(None) is None


# --------------------------------------------------------------------------- #
# pacing
# --------------------------------------------------------------------------- #

def test_pace_takes_a_long_break_on_the_interval(monkeypatch: pytest.MonkeyPatch):
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    pace = Pace(min_delay=8, max_delay=8, break_every=2, break_seconds=90, rng=random.Random(0))
    assert pace.pause(0) == 8
    assert pace.pause(2) == 90
    assert slept == [8, 90]


def test_pace_stays_within_bounds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    pace = Pace(min_delay=5, max_delay=9, break_every=0, rng=random.Random(1))
    assert all(5 <= pace.pause(i) <= 9 for i in range(20))


# --------------------------------------------------------------------------- #
# price
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected",
    [
        ("US$ 19.00", (19.0, "USD")),
        ("£8.50", (8.5, "GBP")),
        ("€ 1,234.50", (1234.5, "EUR")),
        ("C$ 12", (12.0, "CAD")),
        (None, (None, None)),
        ("Ask seller", (None, None)),
    ],
)
def test_parse_price(text, expected):
    assert parse_price(text) == expected


# --------------------------------------------------------------------------- #
# block detection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("status,expected", [(403, "http_403"), (429, "http_429"), (503, "http_503")])
def test_detect_block_reports_throttle_statuses(status: int, expected: str):
    assert detect_block("<html>ok</html>", status) == expected


def test_detect_block_matches_challenge_page():
    assert detect_block("<h1>Please verify you are a human</h1>", 200) == "verify you are a human"


def test_detect_block_passes_normal_page():
    assert detect_block("<html><h1>Art of Computer Programming</h1></html>", 200) is None


# --------------------------------------------------------------------------- #
# search page parsing
# --------------------------------------------------------------------------- #

SEARCH_HTML = """
<html><head><title>Isbn: 0201853949 - results</title></head><body>
<span data-test-id="result-count">(6 results)</span>
<ul data-test-id="srp-search-results-list">
  <li data-test-id="listing-item-32510370781" data-csa-c-item-id="32510370781">
    <figure data-test-id="listing-image">
      <img src="https://pictures.abebooks.com/isbn/9780201853940-us._SL300_.jpg"
           data-csa-c-image-type="stock-image">
    </figure>
    <a href="/Art-Computer-Programming/32510370781/bd" data-test-id="listing-title-link">
      <span data-test-id="listing-title">Art of Computer Programming</span>
    </a>
    <span data-test-id="listing-author">Knuth, Donald E.</span>
    <span data-test-id="star-rating">4-star seller</span>
    <a href="/Greener-Books-London/52838368/sf" data-test-id="listing-seller-link">Greener Books</a>
    <span data-test-id="listing-condition">Used - Very good</span>
    <span data-test-id="listing-price">US$ 19.00</span>
    <span data-test-id="listing-quantity">Quantity: 1 available</span>
  </li>
</ul>
</body></html>
"""


def test_parse_search_page_extracts_a_listing():
    parsed = parse_search_page(SEARCH_HTML, "https://www.abebooks.com/servlet/SearchResults?isbn=0201853949")
    assert parsed["listing_count"] == 1
    assert parsed["result_count_text"] == "(6 results)"
    listing = parsed["listings"][0]
    assert listing["listing_id"] == "32510370781"
    assert listing["listing_title"] == "Art of Computer Programming"
    assert listing["listing_authors"] == "Knuth, Donald E."
    assert listing["seller_name"] == "Greener Books"
    assert listing["condition"] == "Used - Very good"
    assert listing["price"] == 19.0
    assert listing["currency"] == "USD"
    assert listing["availability_quantity"] == 1
    assert listing["seller_rating"] == "4-star seller"
    assert listing["stock_image"] is True
    assert listing["listing_url"] == "https://www.abebooks.com/Art-Computer-Programming/32510370781/bd"
    assert listing["seller_url"].startswith("https://www.abebooks.com/Greener-Books-London/")


def test_parse_search_page_handles_no_results():
    parsed = parse_search_page("<html><title>No results</title></html>", "https://example/search")
    assert parsed["listing_count"] == 0
    assert parsed["listings"] == []


# --------------------------------------------------------------------------- #
# book detail parsing
# --------------------------------------------------------------------------- #

DETAIL_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Book","name":"Art of Computer Programming",
 "isbn":"9780201853940","publisher":{"@type":"Organization","name":"Addison Wesley"},
 "bookFormat":"https://schema.org/Paperback","datePublished":"2005","inLanguage":"English",
 "image":"https://pictures.abebooks.com/isbn/9780201853940-us._SL300_.jpg"}
</script>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","description":"**SHIPPED FROM UK** A treatise."}
</script>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"BookStore","name":"Greener Books",
 "url":"https://www.abebooks.com/Greener-Books-London/52838368/sf","description":"An enterprise."}
</script>
</head><body>
<h1 data-test-id="listing-title">Art of Computer Programming</h1>
<span data-test-id="listing-author">Knuth, Donald E.</span>
<div data-test-id="listing-publisher">Published by Addison Wesley, 2005</div>
<div data-test-id="listing-isbn-link">0201853949 / 9780201853940</div>
<div data-test-id="listing-language">Language: English</div>
<div data-test-id="goodreads-rating">4.54 46 ratings by Goodreads</div>
<div data-test-id="listing-attributes">
  <ul><li aria-label="Softcover">Softcover</li><li aria-label="Used">Used</li></ul>
</div>
<div data-test-id="sf-seller-since">AbeBooks seller since October 31, 2007</div>
</body></html>
"""


def test_parse_book_page_reads_dom_and_json_ld():
    parsed = parse_book_page(DETAIL_HTML, "https://www.abebooks.com/x/1/bd")
    assert parsed["title"] == "Art of Computer Programming"
    assert parsed["authors"] == "Knuth, Donald E."
    assert parsed["isbn10"] == "0201853949"
    assert parsed["isbn13"] == "9780201853940"
    assert parsed["publisher"] == "Addison Wesley"
    assert parsed["publication_year"] == "2005"
    assert parsed["language"] == "English"
    assert parsed["binding"] == "Paperback"
    assert parsed["condition"] == "Softcover, Used"
    assert parsed["goodreads_rating"] == "4.54"
    assert parsed["goodreads_rating_count"] == "46"
    assert "treatise" in parsed["synopsis_text"]
    assert parsed["seller_name"] == "Greener Books"
    assert parsed["seller_since"] == "October 31, 2007"


def test_parse_book_page_survives_a_bare_page():
    parsed = parse_book_page("<html><title>Item no longer available</title></html>", "https://x/y")
    assert parsed["isbn13"] is None
    assert parsed["publisher"] is None
    assert parsed["binding"] is None
    assert parsed["seller_name"] is None


BIBLIO_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Book","name":"Compilers",
 "publisher":{"@type":"Organization","name":"Addison Wesley"},"datePublished":"2005",
 "bookFormat":"https://schema.org/Paperback","inLanguage":"English"}
</script>
</head><body>
<h1 data-test-id="listing-title">Compilers</h1>
<div data-test-id="bibliographic-details-edition">First Edition.</div>
<div data-test-id="bibliographic-details-binding">Hardcover</div>
<div data-test-id="bibliographic-details-publishyear">1985</div>
<div data-test-id="bibliographic-details-publisher">Addison-Wesley</div>
<div data-test-id="bibliographic-details-language">English</div>
<div data-test-id="description-text">The item might be beaten up but readable.</div>
<div data-test-id="about-description">We are an independent online bookseller.</div>
<div data-test-id="about-the-title">Christine Hofmeister is a project manager.</div>
<div data-test-id="bibliographic-details-dustjacket">Yes</div>
<span data-test-id="vendor-listing-id">Seller Inventory # RWARE0000058606</span>
<span data-test-id="buybox-item-shipping-price">US$ 6.00 shipping</span>
<span data-test-id="sf-address-line">Tokyo, TKY, Japan</span>
<span data-test-id="star-rating">2-star seller</span>
<span data-test-id="listing-image-type-label">Stock Image</span>
</body></html>
"""


def test_parse_book_page_prefers_the_bibliographic_block():
    parsed = parse_book_page(BIBLIO_HTML, "https://www.abebooks.com/x/1/bd")
    # the block describes this copy; the JSON-LD only describes the edition
    assert parsed["edition_number"] == "First Edition."
    assert parsed["binding"] == "Hardcover"
    assert parsed["publication_year"] == "1985"
    assert parsed["publisher"] == "Addison-Wesley"
    assert parsed["vendor_description"] == "The item might be beaten up but readable."
    assert parsed["dust_jacket"] == "Yes"


def test_parse_book_page_reads_the_seller_and_shipping_columns():
    parsed = parse_book_page(BIBLIO_HTML, "https://www.abebooks.com/x/1/bd")
    assert parsed["seller_inventory_no"] == "RWARE0000058606"
    assert parsed["shipping_price"] == 6.0
    assert parsed["shipping_currency"] == "USD"
    assert parsed["seller_location"] == "Tokyo, TKY, Japan"
    assert parsed["seller_city"] == "Tokyo"
    assert parsed["seller_region"] == "TKY"
    assert parsed["seller_country"] == "Japan"
    assert parsed["seller_rating"] == "2-star seller"
    assert parsed["stock_image"] is True


def test_parse_book_page_does_not_mistake_the_seller_blurb_for_an_author_bio():
    parsed = parse_book_page(BIBLIO_HTML, "https://www.abebooks.com/x/1/bd")
    # the bio comes from "About this title", never from the seller's own blurb
    assert parsed["about_author_text"] == "Christine Hofmeister is a project manager."
    assert "independent online bookseller" not in parsed["about_author_text"]
    assert "independent online bookseller" not in (parsed["synopsis_text"] or "")


def test_book_is_empty_flags_an_unhydrated_page():
    bare = parse_book_page("<html><title>Art of Computer Programming</title></html>", "https://x/y")
    assert book_is_empty(bare) is True
    assert book_is_empty({"title": "Something"}) is False
    assert book_is_empty({"isbn13": "9780201853940"}) is False


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #

def test_fetched_isbns_spends_only_what_a_rerun_cannot_improve(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    out.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"isbn": "1", "status": "ok"},
                {"isbn": "2", "status": "error"},
                {"isbn": "3", "status": "blocked"},
                {"isbn": "4", "status": "degraded"},
                {"isbn": "5", "status": "no_listings"},
            ]
        ),
        encoding="utf-8",
    )
    # only "ok" and "no_listings" are final. A degraded page is a transient
    # site-side state -- the same URLs served full pages minutes later -- so
    # those ISBNs must stay queued or the dataset loses books for nothing.
    assert fetched_isbns(out) == {"1", "5"}


def test_fetched_isbns_missing_file_is_empty(tmp_path: Path):
    assert fetched_isbns(tmp_path / "nope.jsonl") == set()


def test_fetched_isbns_ignores_truncated_lines(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    out.write_text('{"isbn": "1", "status": "ok"}\n{"isbn": "2", "stat', encoding="utf-8")
    assert fetched_isbns(out) == {"1"}


# --------------------------------------------------------------------------- #
# degrade circuit-breaker
# --------------------------------------------------------------------------- #

class _ThrottledSession:
    """Serves a normal search page but never a hydrated detail page."""

    def __init__(self, search_html: str, detail_html: str):
        self.search_html = search_html
        self.detail_html = detail_html

    def fetch(self, url: str, wait_for: str | None = None):
        return (self.search_html if "SearchResults" in url else self.detail_html, 200)


def test_collect_stops_once_degradation_comes_in_a_run(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    pace = Pace(min_delay=0, max_delay=0, break_every=0, rng=random.Random(0))
    session = _ThrottledSession(SEARCH_HTML, "<html><title>unhydrated</title></html>")

    stats = collect(["1", "2", "3", "4", "5"], out, pace, session, tmp_path / "html", True,
                    degrade_cooldown=0, max_consecutive_degraded=2)

    assert stats["stopped_early"] is True
    assert stats["degraded"] == 2
    # the last three were never requested, so they stay available -- and so do
    # the two degraded ones, whose pages may hydrate on a later attempt
    assert fetched_isbns(out) == set()
    # one unhydrated sample is kept, so the episode can be audited later against
    # the same URL once it works again
    assert (tmp_path / "html" / "1_degraded.html").exists()


def test_collect_keeps_going_when_degradation_is_isolated(tmp_path: Path):
    class _Flaky(_ThrottledSession):
        def __init__(self):
            super().__init__(SEARCH_HTML, "<html><title>unhydrated</title></html>")
            self.calls = 0

        def fetch(self, url: str, wait_for: str | None = None):
            if "SearchResults" in url:
                return (SEARCH_HTML, 200)
            self.calls += 1
            # every record's second listing hydrates, so nothing runs
            return ("<html><h1 data-test-id='listing-title'>Art</h1>"
                    "<div data-test-id='listing-isbn-link'>0201853949 / 9780201853940</div></html>", 200)

    out = tmp_path / "out.jsonl"
    pace = Pace(min_delay=0, max_delay=0, break_every=0, rng=random.Random(0))
    stats = collect(["1", "2", "3"], out, pace, _Flaky(), tmp_path / "html", True,
                    degrade_cooldown=0, max_consecutive_degraded=2)
    assert stats["stopped_early"] is False
    assert stats["ok"] == 3

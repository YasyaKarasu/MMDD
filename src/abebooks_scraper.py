"""Low-rate browser collector for AbeBooks book pages.

Drives a real Chromium through Playwright at human pace to collect the fields
described in ``docs/abebooks_small_multimodal_dataset_plan.zh-CN.md``.

This is deliberately *not* a stealth client. It does not forge a user-agent,
patch bot-detection surfaces, or solve CAPTCHAs; when the site signals that it
has stopped serving pages the run halts and records why. The only concession to
politeness is time: one page at a time, randomised human-length pauses, and a
longer break every few records.

The consent and risk record for this collection lives in the "风险、许可与已记录
的决定" section of that plan document.

Requires ``playwright`` plus ``python -m playwright install chromium``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import BrowserContext, Page, sync_playwright

BASE = "https://www.abebooks.com"
SEARCH_URL = "https://www.abebooks.com/servlet/SearchResults?isbn={isbn}"
LISTING_CARD = "li[data-test-id^='listing-item-']"
DETAIL_READY = "[data-test-id='listing-title']"

# How many distinct listings to try before giving a book up as degraded.
DETAIL_ATTEMPTS = 3

# Statuses where asking again cannot change the answer, so a rerun must skip
# them: "ok" already has the data, and a search with no listings on AbeBooks
# will still have none.
#
# "degraded" is deliberately NOT here. It was, on the theory that an unhydrated
# page is a sticky per-URL property. That theory is wrong: on 2026-09-13 three
# detail URLs came back unhydrated at 14:02:26 and all three served the full
# page again by 14:15, untouched. Degradation is a transient site-side state, so
# marking those ISBNs spent threw away books for a condition that had already
# cleared. "error" and "blocked" are retryable for the same reason.
SPENT_STATUSES = ("ok", "no_listings")

# Phrases AbeBooks serves instead of a page when it stops answering a client.
BLOCK_MARKERS = (
    "are you a robot",
    "unusual traffic",
    "captcha",
    "access denied",
    "request blocked",
    "verify you are a human",
)

CURRENCY_SYMBOLS = {"US$": "USD", "$": "USD", "£": "GBP", "€": "EUR", "C$": "CAD", "AU$": "AUD"}


class BlockedError(RuntimeError):
    """Raised when AbeBooks stops serving pages; the run halts rather than retries."""


# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #

def read_isbns(path: Path, limit: int | None = None) -> list[str]:
    """Distinct normalised ISBNs from the ``source, isbn, title, author`` TSV."""
    seen: set[str] = set()
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        isbn = re.sub(r"[^0-9Xx]", "", parts[1]).upper()
        if isbn and isbn not in seen:
            seen.add(isbn)
            out.append(isbn)
            if limit and len(out) >= limit:
                break
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# pacing
# --------------------------------------------------------------------------- #

@dataclass
class Pace:
    """Human-length pauses between page loads."""

    min_delay: float = 8.0
    max_delay: float = 20.0
    break_every: int = 8
    break_seconds: float = 90.0
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()

    def pause(self, index: int) -> float:
        """Sleep before record ``index``; returns how long was slept."""
        if index and self.break_every and index % self.break_every == 0:
            delay = self.break_seconds
        else:
            delay = self.rng.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)
        return delay


# --------------------------------------------------------------------------- #
# proxy
# --------------------------------------------------------------------------- #

def resolve_proxy(explicit: str | None) -> dict[str, str] | None:
    """Playwright proxy dict from an explicit URL or the ambient *_proxy vars."""
    url = explicit or next(
        (
            os.environ[name]
            for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy")
            if os.environ.get(name)
        ),
        None,
    )
    if not url:
        return None
    # Chromium accepts socks5:// but not the socks5h:// ("remote DNS") spelling.
    return {"server": re.sub(r"^socks5h://", "socks5://", url.strip())}


# --------------------------------------------------------------------------- #
# page parsing (pure functions over HTML, so they stay offline-testable)
# --------------------------------------------------------------------------- #

def field(node, test_id: str) -> str | None:
    """Text of the ``data-test-id`` descendant, or None when absent/empty."""
    found = node.select_one(f"[data-test-id='{test_id}']")
    if not found:
        return None
    return found.get_text(" ", strip=True) or None


def parse_price(text: str | None) -> tuple[float | None, str | None]:
    """``'US$ 19.00'`` -> ``(19.0, 'USD')``."""
    if not text:
        return None, None
    match = re.search(r"([0-9][0-9,]*\.?[0-9]*)", text)
    if not match:
        return None, None
    symbol = text[: match.start()].strip()
    return float(match.group(1).replace(",", "")), CURRENCY_SYMBOLS.get(symbol, symbol or None)


def json_ld_blocks(soup: BeautifulSoup) -> list[dict]:
    out: list[dict] = []
    for node in soup.select("script[type='application/ld+json']"):
        try:
            data = json.loads(node.string or "")
        except (TypeError, ValueError):
            continue
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict):
                out.append(item)
    return out


def ld_of_type(blocks: list[dict], *types: str) -> dict:
    return next((block for block in blocks if block.get("@type") in types), {})


def parse_search_page(html: str, url: str) -> dict:
    """Candidate listings from an ISBN search page (``book_listing`` in the plan)."""
    soup = BeautifulSoup(html, "html.parser")
    listings = []
    for card in soup.select(LISTING_CARD):
        link = card.select_one("a[href*='/bd']")
        seller_link = card.select_one("[data-test-id='listing-seller-link']")
        image = card.select_one("img")
        price, currency = parse_price(field(card, "listing-price"))
        quantity = re.search(r"(\d+)", field(card, "listing-quantity") or "")
        listings.append(
            {
                "listing_id": card.get("data-csa-c-item-id")
                or (card.get("data-test-id") or "").removeprefix("listing-item-"),
                "listing_url": urljoin(BASE, link["href"]) if link else None,
                "listing_title": field(card, "listing-title"),
                "listing_authors": field(card, "listing-author"),
                "seller_name": seller_link.get_text(" ", strip=True) if seller_link else None,
                "seller_url": urljoin(BASE, seller_link["href"]) if seller_link else None,
                "condition": field(card, "listing-condition"),
                "price": price,
                "currency": currency,
                "availability_quantity": int(quantity.group(1)) if quantity else None,
                "seller_rating": field(card, "star-rating"),
                "image_url": image.get("src") if image else None,
                "stock_image": image.get("data-csa-c-image-type") == "stock-image" if image else None,
            }
        )
    return {
        "source_url": url,
        "result_count_text": field(soup, "result-count"),
        "listing_count": len(listings),
        "listings": listings,
    }


def parse_book_page(html: str, url: str) -> dict:
    """Edition and seller fields from a listing detail page (``book_edition``)."""
    soup = BeautifulSoup(html, "html.parser")
    blocks = json_ld_blocks(soup)
    book = ld_of_type(blocks, "Book", "Product")
    store = ld_of_type(blocks, "BookStore")

    isbn_text = field(soup, "listing-isbn-link") or ""
    isbn10, _, isbn13 = (part.strip() for part in isbn_text.partition("/"))

    # The bibliographic block describes this copy, so prefer it over the
    # JSON-LD, which describes the edition in general.
    biblio = {key: field(soup, f"bibliographic-details-{key}")
              for key in ("edition", "binding", "language", "publisher", "publishyear",
                          "condition", "dustjacket", "dimensions", "itemweight", "series")}

    publisher_text = field(soup, "listing-publisher") or ""
    published = re.match(r"Published by (?P<publisher>.+?), (?P<year>\d{4})\s*$", publisher_text)
    if published:
        publisher, year = published.group("publisher"), published.group("year")
    else:
        raw_publisher = book.get("publisher")
        publisher = raw_publisher.get("name") if isinstance(raw_publisher, dict) else raw_publisher
        year = book.get("datePublished")
    publisher = biblio["publisher"] or publisher
    year = biblio["publishyear"] or year

    rating_text = field(soup, "goodreads-rating") or ""
    rating_match = re.match(r"(?P<value>[0-9.]+)\s+(?P<count>\d+)", rating_text)
    rating = rating_match.groupdict() if rating_match else {}

    language = (field(soup, "listing-language") or "").removeprefix("Language:").strip()
    binding = (book.get("bookFormat") or "").removeprefix("https://schema.org/") or None
    attributes = [item.get_text(strip=True) for item in soup.select("[data-test-id='listing-attributes'] li")]

    # Columns for the plan's seller/book_listing tables that only the detail
    # page carries. Each is left None when the seller did not supply it.
    shipping_price, shipping_currency = parse_price(field(soup, "buybox-item-shipping-price"))
    inventory = field(soup, "vendor-listing-id") or ""
    address = [part.strip() for part in (field(soup, "sf-address-line") or "").split(",") if part.strip()]
    city, region, country = address[:3] if len(address) >= 3 else (None, None, None)
    image_label = (field(soup, "listing-image-type-label") or "").strip().lower()

    return {
        "source_url": url,
        "title": field(soup, "listing-title") or book.get("name"),
        "authors": field(soup, "listing-author"),
        "isbn10": isbn10 or None,
        "isbn13": isbn13 or book.get("isbn"),
        "publisher": publisher,
        "publication_year": year,
        "language": biblio["language"] or language or book.get("inLanguage"),
        "binding": biblio["binding"] or binding,
        "edition_number": biblio["edition"],
        "series": biblio["series"],
        "dust_jacket": biblio["dustjacket"],
        "dimensions": biblio["dimensions"],
        "item_weight": biblio["itemweight"],
        "condition": biblio["condition"] or ", ".join(attributes) or None,
        "catalogue_image_url": book.get("image"),
        "stock_image": image_label.startswith("stock") if image_label else None,
        "synopsis_text": ld_of_type(blocks, "Product").get("description"),
        # about-description is the seller's blurb about themselves, not an
        # author bio; the bio lives in the "About this title" section, so the
        # two are read from different nodes on purpose.
        "about_author_text": field(soup, "about-the-title"),
        "vendor_description": field(soup, "description-text"),
        "goodreads_rating": rating.get("value"),
        "goodreads_rating_count": rating.get("count"),
        "seller_name": store.get("name") or field(soup, "sf-name"),
        "seller_url": store.get("url"),
        "seller_since": (field(soup, "sf-seller-since") or "").removeprefix("AbeBooks seller since ").strip() or None,
        "seller_description": store.get("description"),
        "seller_location": field(soup, "sf-address-line"),
        "seller_city": city,
        "seller_region": region,
        "seller_country": country,
        "seller_rating": field(soup, "star-rating"),
        "seller_terms": field(soup, "policy-sales-terms-text"),
        "shipping_terms": field(soup, "policy-shipping-terms-text"),
        "shipping_price": shipping_price,
        "shipping_currency": shipping_currency,
        "seller_inventory_no": inventory.removeprefix("Seller Inventory #").strip() or None,
    }


def book_is_empty(book: dict) -> bool:
    """True when a detail page parsed to nothing, i.e. it came back unhydrated."""
    return not book.get("title") and not book.get("isbn13")


def detect_block(html: str, status: int | None) -> str | None:
    """Return a reason string when the response looks like a refusal."""
    if status in (403, 429, 503):
        return f"http_{status}"
    lowered = html.casefold()
    for marker in BLOCK_MARKERS:
        if marker in lowered:
            return marker
    return None


# --------------------------------------------------------------------------- #
# browser
# --------------------------------------------------------------------------- #

class BrowserSession:
    """A single Chromium instance reused across the whole run."""

    def __init__(self, proxy: dict[str, str] | None, headless: bool, timeout_ms: float,
                 user_data_dir: Path | str | None, user_agent: str | None = None):
        self.proxy = proxy
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.user_data_dir = Path(user_data_dir) if user_data_dir is not None else None
        self.user_agent = user_agent
        self._pw = None
        self._context: BrowserContext | None = None

    def __enter__(self) -> "BrowserSession":
        self._pw = sync_playwright().start()
        # channel="chromium" selects the full browser build; without it Playwright
        # reaches for the separate chrome-headless-shell download.
        launch = {"headless": self.headless, "proxy": self.proxy, "channel": "chromium"}
        context: dict[str, object] = {"locale": "en-US", "timezone_id": "America/New_York"}
        if self.user_agent:
            # This overrides the User-Agent header and nothing else. The browser
            # still sends sec-ch-ua* client hints describing its real version, so
            # an old UA string ends up contradicted by its own request headers.
            context["user_agent"] = self.user_agent
        if self.user_data_dir is not None:
            # Off by default. Carrying cookies between runs does NOT make the
            # site treat us as a returning visitor -- it makes it serve the
            # unhydrated detail page instead (observed 2026-09-13). Only pass
            # this when debugging something that genuinely needs stored state.
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            self._context = self._pw.chromium.launch_persistent_context(
                str(self.user_data_dir), **context, **launch
            )
        else:
            self._context = self._pw.chromium.launch(**launch).new_context(**context)
        self._context.set_default_timeout(self.timeout_ms)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._context is not None:
            self._context.close()
        if self._pw is not None:
            self._pw.stop()

    def fetch(self, url: str, wait_for: str | None = None) -> tuple[str, int | None]:
        """Load ``url`` and return (html, status). Raises BlockedError on refusal."""
        page: Page = self._context.new_page()
        try:
            response = page.goto(url, wait_until="domcontentloaded")
            status = response.status if response else None
            # AbeBooks renders listings client-side, so domcontentloaded alone
            # captures a half-built page. Wait for the network to go quiet, then
            # for the container we care about if the caller named one.
            try:
                page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 20_000))
            except Exception:
                pass
            if wait_for:
                try:
                    page.wait_for_selector(wait_for, timeout=min(self.timeout_ms, 15_000))
                except Exception:
                    pass  # a missing container is the caller's to interpret
            page.mouse.wheel(0, random.randint(400, 1200))
            time.sleep(random.uniform(0.8, 2.2))
            html = page.content()
        finally:
            page.close()
        reason = detect_block(html, status)
        if reason:
            raise BlockedError(f"{reason} at {url}")
        return html, status


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def snapshot(html: str, html_dir: Path, name: str) -> tuple[str, str]:
    """Persist raw HTML for audit; returns (path, sha256)."""
    html_dir.mkdir(parents=True, exist_ok=True)
    path = html_dir / f"{name}.html"
    path.write_text(html, encoding="utf-8")
    return str(path), hashlib.sha256(html.encode("utf-8")).hexdigest()


def fetched_isbns(output: Path) -> set[str]:
    """ISBNs whose source URL is already spent and must not be requested again."""
    if not output.exists():
        return set()
    done: set[str] = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("status") in SPENT_STATUSES:
            done.add(record["isbn"])
    return done


def collect(isbns: list[str], output: Path, pace: Pace, session: BrowserSession,
            html_dir: Path, detailed: bool, degrade_cooldown: float = 0.0,
            max_consecutive_degraded: int = 3) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    done = fetched_isbns(output)
    stats = {"requested": len(isbns), "already_done": len(done), "ok": 0, "degraded": 0,
             "no_listings": 0, "blocked": 0, "errors": 0, "stopped_early": False}
    consecutive_degraded = 0

    with output.open("a", encoding="utf-8") as fh:
        for index, isbn in enumerate(isbns):
            if isbn in done:
                continue
            if stats["ok"] + stats["already_done"] > 0:
                pace.pause(stats["ok"] + stats["already_done"])

            url = SEARCH_URL.format(isbn=quote_plus(isbn))
            record: dict = {"isbn": isbn, "retrieved_at": now_iso(), "source_url": url}
            try:
                html, status = session.fetch(url, wait_for=LISTING_CARD)
                record["http_status"] = status
                record["search"] = parse_search_page(html, url)
            except BlockedError as exc:
                record.update(status="blocked", error=str(exc))
                stats["blocked"] += 1
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{index + 1}/{len(isbns)}] {isbn} BLOCKED: {exc} -- stopping run")
                return stats
            except Exception as exc:  # network/JS failures are recorded, not retried
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                stats["errors"] += 1
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{index + 1}/{len(isbns)}] {isbn} error: {exc}")
                continue

            record["html_path"], record["html_sha256"] = snapshot(html, html_dir, isbn)

            if detailed and not record["search"]["listings"]:
                # AbeBooks has nothing for this ISBN, so there is no detail page
                # to open. Kept apart from "ok" so it does not dilute per-field
                # coverage, and spent so a rerun does not ask again.
                record.update(status="no_listings")
                stats["no_listings"] += 1
                consecutive_degraded = 0
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{index + 1}/{len(isbns)}] {isbn} no listings on AbeBooks")
                continue

            if detailed and record["search"]["listings"]:
                try:
                    for attempt, listing in enumerate(record["search"]["listings"][:DETAIL_ATTEMPTS]):
                        if attempt:
                            pace.pause(0)
                        detail_url = listing["listing_url"]
                        detail_html, _ = session.fetch(detail_url, wait_for=DETAIL_READY)
                        book = parse_book_page(detail_html, detail_url)
                        # An unhydrated detail page means the site is serving the
                        # lightweight variant; it says nothing about this listing.
                        # The state is transient (see SPENT_STATUSES), so try a
                        # different listing, and if they are all unhydrated back
                        # off in the caller. Keep one sample so the episode can be
                        # audited afterwards -- without it there is nothing to
                        # compare against when the same URL works later.
                        if book_is_empty(book):
                            if "degraded_html_path" not in record:
                                record["degraded_url"] = detail_url
                                (record["degraded_html_path"],
                                 record["degraded_html_sha256"]) = snapshot(
                                    detail_html, html_dir, f"{isbn}_degraded")
                            continue
                        record["book"] = book
                        record["book_html_path"], record["book_html_sha256"] = snapshot(
                            detail_html, html_dir, f"{isbn}_detail"
                        )
                        break
                    else:
                        record.update(status="degraded")
                        stats["degraded"] += 1
                        consecutive_degraded += 1
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()
                        print(f"[{index + 1}/{len(isbns)}] {isbn} all {DETAIL_ATTEMPTS} detail "
                              f"listings came back degraded "
                              f"({consecutive_degraded}/{max_consecutive_degraded} in a row)")
                        # Degradation arrives in clusters and also hits URLs we
                        # have never requested, so a run of them means the site
                        # has stopped hydrating detail pages for now. Continuing
                        # would just mark more records degraded, so stop and let
                        # the operator decide. Nothing is lost permanently: a
                        # rerun retries every degraded ISBN.
                        if consecutive_degraded >= max_consecutive_degraded:
                            stats["stopped_early"] = True
                            print(f"STOP: {consecutive_degraded} consecutive degraded records -- "
                                  f"AbeBooks is throttling. {len(isbns) - index - 1} ISBNs left "
                                  f"untouched; wait for the throttle to clear, then rerun.")
                            return stats
                        if degrade_cooldown > 0:
                            time.sleep(degrade_cooldown)
                        continue
                except BlockedError as exc:
                    record.update(status="blocked", error=str(exc))
                    stats["blocked"] += 1
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(f"[{index + 1}/{len(isbns)}] {isbn} BLOCKED on detail: {exc} -- stopping run")
                    return stats
                except Exception as exc:
                    record["book"] = {"error": f"{type(exc).__name__}: {exc}"}

            record["status"] = "ok"
            stats["ok"] += 1
            consecutive_degraded = 0
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{index + 1}/{len(isbns)}] {isbn} ok ({record['search']['listing_count']} listings)")

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("book.txt"))
    p.add_argument("--output", type=Path, default=Path("output/abebooks_pilot.jsonl"))
    p.add_argument("--html-dir", type=Path, default=Path("output/abebooks_html"))
    p.add_argument("--limit", type=int, default=50, help="number of ISBNs to seed the queue with")
    p.add_argument("--min-delay", type=float, default=8.0, help="seconds between records (lower bound)")
    p.add_argument("--max-delay", type=float, default=20.0, help="seconds between records (upper bound)")
    p.add_argument("--break-every", type=int, default=8, help="take a longer break every N records")
    p.add_argument("--break-seconds", type=float, default=90.0)
    p.add_argument("--proxy", default=None, help="proxy URL; defaults to the ambient *_proxy vars")
    p.add_argument("--no-proxy", action="store_true", help="ignore proxy settings entirely")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--headed", dest="headless", action="store_false", help="needs a display or xvfb-run")
    p.add_argument("--profile", dest="user_data_dir", type=Path, default=None,
                   help="reuse a persistent browser profile at this path; off by default because "
                        "a dirty profile makes AbeBooks serve unhydrated detail pages")
    p.add_argument("--user-agent", default=None,
                   help="override the browser's User-Agent string; unset by default, and unset is "
                        "the only configuration that has been observed to work. This overrides the "
                        "header only: the browser keeps sending sec-ch-ua* client hints for its "
                        "real version, so a stale UA ends up contradicted by its own request "
                        "headers. A 2026-09-16 attempt to blame the UA was inconclusive -- both "
                        "runs began on the same unfinished record, so the comparison isolated "
                        "nothing")
    p.add_argument("--degrade-cooldown", type=float, default=180.0,
                   help="seconds to idle after a record whose detail pages all came back degraded")
    p.add_argument("--max-consecutive-degraded", type=int, default=3,
                   help="abort the run after this many degraded records in a row; each one spends "
                        "an ISBN, so continuing through a throttle only destroys data")
    p.add_argument("--no-detailed", dest="detailed", action="store_false", default=True,
                   help="collect search results only, skip the detail page")
    p.add_argument("--timeout", type=float, default=45_000, help="per-page timeout in ms")
    p.add_argument("--seed", type=int, default=None, help="fix the pacing RNG for a reproducible run")
    args = p.parse_args()

    proxy = None if args.no_proxy else resolve_proxy(args.proxy)
    pace = Pace(min_delay=args.min_delay, max_delay=args.max_delay, break_every=args.break_every,
                break_seconds=args.break_seconds, rng=random.Random(args.seed))
    isbns = read_isbns(args.input, args.limit)
    print(f"{len(isbns)} ISBNs queued; proxy={proxy or 'direct'}; "
          f"pace={args.min_delay}-{args.max_delay}s, break {args.break_seconds}s every {args.break_every}")

    with BrowserSession(proxy, args.headless, args.timeout, args.user_data_dir,
                        args.user_agent) as session:
        stats = collect(isbns, args.output, pace, session, args.html_dir, args.detailed,
                        args.degrade_cooldown, args.max_consecutive_degraded)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

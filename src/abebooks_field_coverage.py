"""Summarise field coverage across an abebooks pilot jsonl.

Usage: python coverage.py output/abebooks_pilot.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

# Groups mirror the plan doc's tables.
BOOK_FIELDS = [
    "title", "authors", "isbn10", "isbn13", "publisher", "publication_year",
    "language", "binding", "edition_number", "series", "dust_jacket",
    "dimensions", "item_weight", "condition", "goodreads_rating",
    "goodreads_rating_count", "catalogue_image_url", "stock_image",
]
TEXT_FIELDS = ["synopsis_text", "about_author_text", "vendor_description"]
SELLER_FIELDS = [
    "seller_name", "seller_url", "seller_since", "seller_location", "seller_city",
    "seller_region", "seller_country", "seller_rating", "seller_terms",
    "shipping_terms", "seller_description",
]
LISTING_FIELDS = [
    "listing_id", "listing_title", "listing_authors", "seller_name", "seller_url",
    "condition", "price", "currency", "availability_quantity", "seller_rating",
    "stock_image", "listing_url",
]
DETAIL_ONLY_FIELDS = ["seller_inventory_no", "shipping_price", "shipping_currency"]


def present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, dict)) and not value:
        return False
    return True


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "output/abebooks_pilot.jsonl")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue

    status = Counter(r.get("status") for r in records)
    print(f"{path}: {len(records)} records")
    for name, count in status.most_common():
        print(f"  {name:<10} {count:>4}  ({count / len(records):.0%})")

    ok = [r for r in records if r.get("status") == "ok"]
    with_detail = [r for r in ok if r.get("book")]
    degraded = [r for r in records if r.get("status") == "degraded"]
    print(f"\nok records: {len(ok)}  (with a parsed detail page: {len(with_detail)})")
    print(f"records with >=1 search listing: "
          f"{sum(1 for r in records if r.get('search', {}).get('listing_count'))}")
    if degraded:
        print(f"degraded records that still carry search data: "
              f"{sum(1 for r in degraded if r.get('search', {}).get('listings'))}")
    zero = [r["isbn"] for r in ok if not r.get("search", {}).get("listing_count")]
    if zero:
        print(f"ok records with zero listings (no detail page possible): {len(zero)} {zero[:5]}")

    if not with_detail:
        print("\nno parsed detail pages -- nothing to score")
        return

    n = len(with_detail)
    print(f"\n{'field':<26} {'non-null':>8} {'rate':>6}")
    print("-" * 42)
    for label, fields, source in (
        ("edition (book_edition)", BOOK_FIELDS, "book"),
        ("text evidence", TEXT_FIELDS, "book"),
        ("seller (seller)", SELLER_FIELDS, "book"),
        ("listing-only detail columns", DETAIL_ONLY_FIELDS, "book"),
    ):
        print(f"{label}:")
        for f in fields:
            c = sum(1 for r in with_detail if present(r.get(source, {}).get(f)))
            print(f"  {f:<24} {c:>8} {c / n:>5.0%}")
    print("search card (book_listing):")
    for f in LISTING_FIELDS:
        c = sum(1 for r in ok
                if r.get("search", {}).get("listings") and present(r["search"]["listings"][0].get(f)))
        print(f"  {f:<24} {c:>8} {c / len(ok):>5.0%}")

    counts = [r.get("search", {}).get("listing_count", 0) for r in ok]
    if counts:
        print(f"\nlistings per ok record: min={min(counts)} max={max(counts)} "
              f"mean={sum(counts) / len(counts):.1f}")
    # The paper's premise is providers disagreeing about a book's authors, so
    # that is the conflict signal worth counting; per-copy condition just varies.
    conflicts = sum(
        1 for r in ok
        if len({l.get("listing_authors") for l in r.get("search", {}).get("listings", [])} - {None}) > 1
    )
    print(f"records whose listings disagree on author list: {conflicts}")
    sellers = [
        len({l.get("seller_name") for l in r.get("search", {}).get("listings", [])} - {None})
        for r in ok if r.get("search", {}).get("listings")
    ]
    if sellers:
        print(f"distinct sellers per record: min={min(sellers)} max={max(sellers)} "
              f"mean={sum(sellers) / len(sellers):.1f}")


if __name__ == "__main__":
    main()

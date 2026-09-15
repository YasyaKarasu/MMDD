#!/usr/bin/env python
"""Build the four AbeBooks tables (book_edition, seller, book_listing, evidence_asset).

Reads the scrape records the AbeBooks collector appends to, and writes the
tables plus their stats, split and manifest.  The build is idempotent and holds
no state, so it can be re-run after any collection run: ISBNs that moved from
degraded to ok are promoted automatically, and records that lost are listed in
``superseded_records.jsonl`` rather than silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

# Running this file directly puts ``scripts_old`` on ``sys.path``, not ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abebooks_tables import build_tables, load_records, split_book_ids
from mmdd_dataset.utils import write_json, write_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_isbns(path: Path) -> list[str]:
    """Distinct normalised ISBNs from the ``source, isbn, title, author`` TSV.

    Mirrors ``abebooks_scraper.read_isbns`` -- imported rather than reimplemented
    would drag Playwright into this module's import graph, and the build already
    cross-checks the two by asserting every record's page ISBN-13 matches its
    seed ISBN.
    """
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
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    records = load_records(input_path)
    seed = _seed_isbns(Path(args.seed)) if args.seed else None
    built = build_tables(records, seed, prefer_seller_image=args.prefer_seller_image,
                         max_chars=args.max_cell_chars)

    table_names = ("book_edition", "seller", "book_listing", "evidence_asset")
    counts = {name: write_jsonl(output_dir / f"{name}.jsonl", built[name])
              for name in table_names}
    write_jsonl(output_dir / "unresolved.jsonl", built["unresolved"])
    write_jsonl(output_dir / "superseded_records.jsonl", built["superseded"])

    stats = built["stats"]
    stats["prefer_seller_image"] = args.prefer_seller_image
    stats["max_cell_chars"] = args.max_cell_chars
    write_json(output_dir / "stats.json", stats)
    write_json(output_dir / "splits.json",
               split_book_ids([row["book_id"] for row in built["book_edition"]], args.seed_id))
    write_json(output_dir / "dataset_manifest.json", {
        "format": "mmdd_abebooks_tables_v1",
        "input": {"path": str(input_path), "sha256": _sha256(input_path),
                  "records": len(records)},
        "artifacts": {
            name: {"path": f"{name}.jsonl", "records": count,
                   "sha256": _sha256(output_dir / f"{name}.jsonl")}
            for name, count in counts.items()
        },
        "single_files": {
            "stats": "stats.json",
            "splits": "splits.json",
            "unresolved": "unresolved.jsonl",
            "superseded_records": "superseded_records.jsonl",
        },
        "notes": [
            "title/authors come from the AbeBooks detail page: byte-identical to "
            "the chosen listing's card strings, so a seller's claim rather than an "
            "edition truth.",
            "evidence_asset sha256/local_path are the containing page's for text "
            "assets, and null for image rows, which are url_only until downloaded.",
            "book_id is isbn13 for every ISBN whose checksum validates; see "
            "isbn_scheme for the fallback.",
        ],
    })
    print(f"built {counts['book_edition']} book_edition, {counts['seller']} seller, "
          f"{counts['book_listing']} book_listing, {counts['evidence_asset']} evidence_asset "
          f"({len(built['unresolved'])} unresolved)")
    return stats


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", default="output/abebooks_full.jsonl",
                        help="scrape records (JSONL, append-only, owned by the scraper)")
    result.add_argument("--seed", default="book.txt",
                        help="seed TSV, used only for the coverage denominator and an "
                             "integrity check; pass '' to skip")
    result.add_argument("--output-dir", default="output/abebooks_dataset")
    result.add_argument("--prefer-seller-image", action="store_true",
                        help="rank a card with a real seller photo above the detail "
                             "listing. Changes which columns are populated and rewrites "
                             "selected_reason, so write it to its own --output-dir; the "
                             "two labellings must not be mixed.")
    result.add_argument("--max-cell-chars", type=int, default=1024,
                        help="cap on stored cell text; the untruncated source stays in "
                             "the page snapshot")
    result.add_argument("--seed-id", type=int, default=13,
                        help="seed for the deterministic book_id train/dev/test split")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

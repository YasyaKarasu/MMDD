"""Rebuild a versioned AbeBooks dataset from source fields and existing labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmdd_dataset.abebooks_source_rebuild import BIBLIOGRAPHIC_COLUMNS, rebuild_from_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-columns", nargs="+", default=sorted(BIBLIOGRAPHIC_COLUMNS))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--readable-headers", action="store_true")
    parser.add_argument("--natural-authors", action="store_true")
    parser.add_argument("--contextualize-book-text", action="store_true",
                        help="Prefix every book text asset with its source-row title, keeping the complete original text")
    parser.add_argument("--strip-series-notes", action="store_true",
                        help="Remove parenthetical series metadata from source book titles before query construction")
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument("--group-by-title", action="store_true",
                        help="Repartition all book rows by title-only balanced TF-IDF clustering")
    grouping.add_argument("--group-by-authors", action="store_true",
                         help="Repartition all book rows by author-name-only balanced TF-IDF clustering")
    grouping.add_argument("--group-by-publisher", action="store_true",
                         help="Use one source table per exact original publisher value; preserve every row")
    grouping.add_argument("--title-embeddings", type=Path,
                         help="Group by title_embeddings.npz from encode_abebooks_source_titles.py")
    parser.add_argument("--publisher-min-rows", type=int, default=1,
                        help="With publisher grouping, coalesce smaller publisher groups into one tail table")
    parser.add_argument("--proportional-splits", action="store_true",
                        help="Preserve 80/10/10 for odd holdout sizes; alternate the extra implicit/explicit query")
    parser.add_argument("--source-reference", type=Path,
                        help="Restore original source fields/entities/links only; labels still come from --dataset")
    args = parser.parse_args()
    report = rebuild_from_sources(args.dataset.resolve(), args.output.resolve(),
                                  keep=set(args.keep_columns), seed=args.seed, readable_headers=args.readable_headers,
                                  natural_authors=args.natural_authors,
                                  source_reference=args.source_reference.resolve() if args.source_reference else None,
                                  group_by_title=args.group_by_title, proportional_splits=args.proportional_splits,
                                  strip_series_notes=args.strip_series_notes, group_by_authors=args.group_by_authors,
                                  title_embeddings=args.title_embeddings.resolve() if args.title_embeddings else None,
                                  contextualize_book_text=args.contextualize_book_text,
                                  group_by_publisher=args.group_by_publisher,
                                  publisher_min_rows=args.publisher_min_rows)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"input_hashes", "unretained_approved_facts"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Project source fields into a new AbeBooks copy with unchanged tasks and labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmdd_dataset.abebooks_source_projection import project_source_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-columns", nargs="*", default=[])
    parser.add_argument("--natural-authors", action="store_true",
                        help="Normalize source author display order while preserving complete join keys")
    parser.add_argument("--contextualize-book-text", action="store_true",
                        help="Prepend each text fragment's existing source-book title; preserve its full body")
    parser.add_argument("--remove-assets-json", type=Path,
                        help="JSON list of asset IDs produced by a recorded curation rule")
    args = parser.parse_args()
    removed = set(json.loads(args.remove_assets_json.read_text())) if args.remove_assets_json else set()
    report = project_source_dataset(args.dataset, args.output,
                                    drop_columns=set(args.drop_columns), removed_assets=removed,
                                    natural_authors=args.natural_authors,
                                    contextualize_book_text=args.contextualize_book_text)
    print(json.dumps({k: report[k] for k in ("destination", "queries", "targets", "assets",
                                             "normalized_qrels", "split_counts")}, indent=2))


if __name__ == "__main__":
    main()

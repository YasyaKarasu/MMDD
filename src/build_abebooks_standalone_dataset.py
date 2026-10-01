#!/usr/bin/env python
"""Build a separate, group-split AbeBooks dataset from blind local annotations."""
import argparse
import json
from pathlib import Path

from mmdd_dataset.abebooks_standalone import build_standalone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--historical-reviews", type=Path, required=True)
    parser.add_argument("--publisher-proposals", type=Path)
    parser.add_argument("--publisher-reviews", type=Path)
    parser.add_argument("--implicit-rows", type=int, default=5)
    parser.add_argument("--explicit-rows", type=int, default=5)
    parser.add_argument("--minimum-recovered-rows", type=int, default=2)
    parser.add_argument("--balanced", action="store_true", help="Keep a 1:1 query-kind ratio within each split")
    args = parser.parse_args()
    report = build_standalone(args.dataset, args.output, args.proposals, args.historical_reviews,
                             implicit_rows=args.implicit_rows, explicit_rows=args.explicit_rows,
                             minimum_recovered_rows=args.minimum_recovered_rows, balanced=args.balanced,
                             publisher_proposals_path=args.publisher_proposals,
                             publisher_reviews_path=args.publisher_reviews)
    print(json.dumps({k: v for k, v in report.items() if k != "input_hashes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

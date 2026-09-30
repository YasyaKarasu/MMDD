"""Rebuild explicit tasks without reusing any historically implicit source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmdd_dataset.abebooks_explicit import regenerate_explicit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    report = regenerate_explicit(args.source.resolve(), args.output.resolve(), args.seed)
    print(json.dumps({k: report[k] for k in ("implicit_queries", "explicit_queries", "implicit_sources",
          "explicit_sources", "shared_sources", "targets", "qrels", "split_counts", "shortfall_by_split")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Add a safe bibliographic context column to single-column AbeBooks queries."""
import argparse
import json
from pathlib import Path

from mmdd_dataset.abebooks_query_context import enrich_query_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = enrich_query_context(args.dataset, args.output)
    print(json.dumps({"queries": report["queries"], "split_counts": report["split_counts"],
                      "query_context": {k: v for k, v in report["query_context"].items() if k != "input_hashes"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

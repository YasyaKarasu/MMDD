#!/usr/bin/env python
"""Copy a standalone AbeBooks dataset and balance query kinds within each split."""
import argparse
import json
from pathlib import Path

from mmdd_dataset.abebooks_rebalance import balance_standalone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    report = balance_standalone(args.dataset, args.output, args.seed)
    print(json.dumps({k: report[k] for k in ("destination", "queries", "split_counts", "groups_per_split",
                     "qrels", "recovery_paths", "balanced_source_unchanged", "preserved_artifacts_unchanged")}, indent=2))


if __name__ == "__main__":
    main()

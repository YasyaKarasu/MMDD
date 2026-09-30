"""Build the four-column-pruned, query-balanced AbeBooks experiment dataset."""
import argparse
import json
from pathlib import Path

from mmdd_dataset.abebooks_rebalance import build_balanced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    report = build_balanced(args.source, args.output, args.seed)
    print(json.dumps({key: report[key] for key in ("after_join_task_removal", "after_balance",
        "split_counts", "queries", "targets", "qrels", "evidence_recoveries", "integrity")}, indent=2))


if __name__ == "__main__":
    main()

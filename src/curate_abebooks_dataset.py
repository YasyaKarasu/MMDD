#!/usr/bin/env python
"""Copy and curate the fixed AbeBooks candidate lake without any model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmdd_dataset.abebooks_curation import curate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-reviews", type=Path, required=True)
    args = parser.parse_args()
    summary = curate_dataset(args.dataset, args.output, args.evidence_reviews)
    print(json.dumps({k: v for k, v in summary.items() if k != "input_hashes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

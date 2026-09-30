#!/usr/bin/env python
"""Run fresh B+IDF fusion from Stage-1/IDF inputs and current bridge scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmdd_stage2.b_plus_idf import run_addendum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idf-scores", type=Path, required=True,
                        help="Frozen selector-independent C50 IDF per-query JSON")
    parser.add_argument("--fresh-scores", type=Path, required=True,
                        help="Current fresh run scores/<arm>/<split>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="SEL_E_FRESH")
    args = parser.parse_args()
    payload = json.loads(args.idf_scores.read_text())
    receipt = run_addendum(payload, args.fresh_scores, args.output, arm=args.arm)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()


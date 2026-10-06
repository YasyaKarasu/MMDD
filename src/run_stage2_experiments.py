#!/usr/bin/env python
"""Prepare isolated R5 arms and analyze results. GPU recovery uses run_stage2.py."""
from __future__ import annotations

import argparse
from pathlib import Path

from mmdd_stage2.experiments import ARMS, audit_recovery, compare, prepare, teacher_ranking, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--stage1-run", type=Path)
    p.add_argument("--z-dir", type=Path)
    p.add_argument("--groups", type=int, default=0, help="source groups per split; 0=all")
    p.add_argument("--splits", nargs="+", choices=("dev", "test"), default=["dev", "test"])
    p = sub.add_parser("compare")
    p.add_argument("--method", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("teacher-ranking")
    p.add_argument("--stage1-run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("audit-recovery")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--review-groups", type=int, default=60)
    p = sub.add_parser("verify")
    p.add_argument("--run", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "compare": compare, "teacher-ranking": teacher_ranking,
     "audit-recovery": audit_recovery, "verify": verify}[command](**args)


if __name__ == "__main__":
    main()

"""Command-line entrypoint for the V4.1 correctness-locked package."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import STAGES, amend_source, prepare, run_formal, smoke, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "validate", "smoke", "all", "amend-source"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--protocol", type=Path, required=True)
        sub.add_argument("--run-root", type=Path, required=True)
    amend = subparsers.choices["amend-source"]
    amend.add_argument("--amendment-id", required=True)
    amend.add_argument("--carry", nargs="+", choices=STAGES, required=True)
    amend.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.protocol, args.run_root)
    elif args.command == "validate":
        validate(args.protocol, args.run_root)
    elif args.command == "smoke":
        smoke(args.protocol, args.run_root)
    elif args.command == "amend-source":
        amend_source(
            args.protocol, args.run_root, amendment_id=args.amendment_id,
            carried_stages=args.carry, reason=args.reason,
        )
    else:
        run_formal(args.protocol, args.run_root)


if __name__ == "__main__":
    main()

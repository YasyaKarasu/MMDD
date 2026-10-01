"""Command-line entrypoint for the Stage-1 CQET pipeline.

Order of operations for a run root:

    lock -> verify-features -> prepare -> validate -> smoke -> all

``lock`` and ``verify-features`` freeze the feature recipe and check it numerically,
``prepare`` builds labels/PCA/provenance, ``validate`` runs the CPU reference tests,
``smoke`` trains every stage on a handful of queries, and ``all`` runs the nine formal
stages followed by the frozen dev/test evaluation. Every path and the GPU identity come
from the protocol file; see ``configs/mmdd_stage1_cqet_protocol.json``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import preflight
from .config import resolve_default_paths
from .pipeline import STAGES, amend_source, prepare, run_formal, smoke, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("lock", "verify-features", "prepare", "validate", "smoke", "all", "amend-source"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--protocol", type=Path, required=True)
        sub.add_argument("--run-root", type=Path, required=True)
    subparsers.choices["verify-features"].add_argument("--device", default="cuda:0")
    amend = subparsers.choices["amend-source"]
    amend.add_argument("--amendment-id", required=True)
    amend.add_argument("--carry", nargs="+", choices=STAGES, required=True)
    amend.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command in ("lock", "verify-features"):
        preflight.configure(resolve_default_paths(args.protocol, args.run_root))
        if args.command == "lock":
            preflight.run_lock()
        else:
            preflight.verify_feature_provenance(args.device)
    elif args.command == "prepare":
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

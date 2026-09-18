"""``python -m mmdd_stage1_clean`` — the CLEAN-R1 Stage-1 entry point.

Spec section 11.  Each subcommand reads only the artifacts of its completed
predecessors in the same run; a failure never silently skips a step.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from . import config as config_module
from .util import CommandReceipt, log_line

SPEC_DIR = Path(__file__).resolve().parent.parent.parent / "mmdd_s1_clean_r1"


def _spec_dir() -> Path:
    override = os.environ.get("MMDD_CLEAN_SPEC_DIR")
    return Path(override) if override else SPEC_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmdd_stage1_clean")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, type=Path)

    p = sub.add_parser("audit-input", help="resolve paths, hash inputs, build raw GT")
    add_config(p)
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("build-objects", help="enumerate the object set and corpus")
    add_config(p)

    p = sub.add_parser("cache", help="frozen one-pass Qwen encoding")
    add_config(p)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-per-modality", type=int, default=None)
    p.add_argument("--limit-tables", type=int, default=None)
    p.add_argument("--limit-evidence", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--range-id", type=int, default=0)
    p.add_argument("--range-count", type=int, default=1)
    p.add_argument("--rank-offset", type=int, default=0)
    p.add_argument("--rank-limit", type=int, default=None)
    p.add_argument("--merge-shards", action="store_true")
    p.add_argument("--only-missing", action="store_true")

    p = sub.add_parser("raw-retrieve", help="raw exact Qwen rankings")
    add_config(p)
    p.add_argument("--split", required=True, choices=["train", "dev", "test"])

    p = sub.add_parser("build-supervision", help="D/E/C/B training packets")
    add_config(p)

    p = sub.add_parser("train-teacher", help="train the shared Teacher")
    add_config(p)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-queries", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--prefetch-workers", type=int, default=4)

    p = sub.add_parser("freeze-teacher", help="select and freeze the best Teacher")
    add_config(p)
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("train-student", help="train one Student arm")
    add_config(p)
    p.add_argument("--arm", required=True, choices=["SUP", "KD"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-queries", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--prefetch-workers", type=int, default=4)

    p = sub.add_parser("freeze-selection", help="lock best-of-run selections")
    add_config(p)

    p = sub.add_parser("evaluate", help="final test evaluation")
    add_config(p)
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--device", default="cuda")
    # Restricting the method set lets the formal table be produced by several
    # processes at once, each on its own device, instead of one long serial run.
    p.add_argument("--methods", default=None,
                   help="comma-separated subset of the formal methods")

    p = sub.add_parser("merge-methods", help="recombine parallel evaluation groups")
    add_config(p)

    p = sub.add_parser("diagnose", help="fixed-pool, strict-EO and resource diagnostics")
    add_config(p)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dev-queries", type=int, default=None)

    p = sub.add_parser("package", help="assemble REPORT.md, COMMANDS.jsonl, MANIFEST")
    add_config(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    cwd = Path.cwd()
    spec_dir = _spec_dir()
    spec = config_module.load_spec_config(spec_dir)
    output_root = (cwd / spec["paths"]["output_root"]).resolve()
    receipt = CommandReceipt(args.command, argv, cwd, output_root / "COMMANDS.jsonl")

    from . import commands

    handler = getattr(commands, f"cmd_{args.command.replace('-', '_')}")
    try:
        result = handler(args, spec, spec_dir, cwd, output_root, receipt)
    except BaseException as error:  # noqa: BLE001 - receipts must record failures
        import traceback

        receipt.finish(
            exit_code=1,
            error=repr(error),
            traceback=traceback.format_exc(),
            spec_dir=str(spec_dir),
        )
        log_line(f"FAILED {args.command}: {error!r}")
        raise
    receipt.finish(exit_code=0, result=result, spec_dir=str(spec_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

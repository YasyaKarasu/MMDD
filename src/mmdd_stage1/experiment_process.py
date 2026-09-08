"""Shared subprocess execution for Stage-1 experiment entrypoints."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path


def run_logged_subprocess(
    command: Sequence[str], log_path: Path, *, root: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            list(command),
            cwd=root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

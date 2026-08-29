#!/usr/bin/env python
"""Render a compact live view of the WDC balance stage.

The builder already publishes durable progress.json telemetry.  This view is
intentionally dependency-free so it can run in a tmux pane while a 200k build
is active.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


PHASES = ("scan_decisions", "select_candidates", "materialize_sources", "verify")


def _gib(value: Any) -> str:
    try:
        return f"{float(value) / (1024**3):.1f} GiB"
    except (TypeError, ValueError):
        return "?"


def _bar(completed: int, total: int, width: int = 36) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    filled = min(width, max(0, width * completed // total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _eta(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(value) or value < 0:
        return "?"
    value = int(value)
    minutes, seconds = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def render(payload: dict[str, Any]) -> str:
    counters = payload.get("counters") or {}
    detail = str(payload.get("detail") or "-")
    completed = int(counters.get("materialization_balance_completed_sources", 0))
    total = int(counters.get("materialization_balance_total_sources", 0))
    selected = int(counters.get("materialization_balance_selected_candidate_count", 0))
    candidates = int(counters.get("materialization_balance_candidate_count", 0))
    subphase = "-"
    for phase in PHASES:
        if phase in detail:
            subphase = phase
            break
    if total and completed >= total:
        subphase = "verify"
    elapsed = float(payload.get("elapsed_seconds") or 0.0)
    rate = completed / elapsed if elapsed > 0 else 0.0
    disk = payload.get("disk") or {}
    lines = [
        "WDC 200k balance pipeline",
        f"stage: {payload.get('stage', '-')} | {subphase} | {detail}",
        f"sources {_bar(completed, total)} {completed:,}/{total:,} "
        f"({100 * completed / total:.1f}% )" if total else "sources [unknown]",
        "flow: " + " -> ".join(
            f"[{('X' if phase == subphase else ' ')}] {phase}"
            for phase in PHASES
        ),
        f"candidates: {candidates:,} scanned | {selected:,} selected | "
        f"{rate:.1f} sources/s | ETA {_eta(payload.get('eta_seconds'))}",
        f"work: {_gib(disk.get('work_bytes'))} | free: {_gib(disk.get('free_bytes'))} | "
        f"reserve: {_gib(disk.get('reserve_bytes'))}",
        f"updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("progress", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.progress.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(f"progress unavailable: {error}")
        return
    if not isinstance(payload, dict):
        print("progress.json is not an object")
        return
    print(render(payload))


if __name__ == "__main__":
    main()

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
STAGES = ("selection", "structural", "sampling", "pages", "asset_planning", "images", "models", "materialize")


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


def _render_balance(payload: dict[str, Any]) -> str:
    counters = payload.get("counters") or {}
    detail = str(payload.get("detail") or "-")
    completed = int(counters.get("materialization_balance_completed_sources", 0))
    total = int(counters.get("materialization_balance_total_sources", 0))
    selected = int(counters.get("materialization_balance_selected_candidate_count", 0))
    candidates = int(counters.get("materialization_balance_candidate_count", 0))
    phase_completed = int(counters.get("materialization_phase_completed", 0))
    phase_total = int(counters.get("materialization_phase_total", 0))
    workers = int(counters.get("materialization_balance_workers", 1))
    prepare_ms = int(counters.get("materialization_balance_prepare_ms", 0))
    write_ms = int(counters.get("materialization_balance_write_ms", 0))
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
        f"phase {_bar(phase_completed, phase_total)} "
        f"{phase_completed:,}/{phase_total:,}" if phase_total else "phase [unknown]",
        "flow: " + " -> ".join(
            f"[{('X' if phase == subphase else ' ')}] {phase}"
            for phase in PHASES
        ),
        f"candidates: {candidates:,} scanned | {selected:,} selected | "
        f"{rate:.1f} sources/s | ETA {_eta(payload.get('eta_seconds'))}",
        f"workers: {workers} | last batch prepare/write: "
        f"{prepare_ms / 1000:.2f}s / {write_ms / 1000:.2f}s",
        f"work: {_gib(disk.get('work_bytes'))} | free: {_gib(disk.get('free_bytes'))} | "
        f"reserve: {_gib(disk.get('reserve_bytes'))}",
        f"updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
    ]
    return "\n".join(lines)


def _render_stage(payload: dict[str, Any]) -> str:
    stage = str(payload.get("stage") or "-")
    detail = str(payload.get("detail") or "-")
    completed = int(payload.get("completed_shards") or 0)
    total = int(payload.get("total_shards") or 0)
    counters = payload.get("counters") or {}
    elapsed = float(payload.get("elapsed_seconds") or 0.0)
    rate = completed / elapsed if elapsed > 0 else 0.0
    disk = payload.get("disk") or {}
    stage_flow = " -> ".join(
        f"[{('X' if name == stage else ' ')}] {name}" for name in STAGES
    )
    live = []
    for key in (
        "structural_tables_completed_live",
        "entities_live",
        "sampling_shards_completed",
        "sampling_shards_total",
    ):
        if key in counters:
            live.append(f"{key.removesuffix('_live').replace('_', ' ')}={int(counters[key]):,}")
    lines = [
        "WDC 200k pipeline",
        f"stage: {stage} | {detail}",
        f"progress {_bar(completed, total)} {completed:,}/{total:,}"
        if total
        else "progress [unknown]",
        "flow: " + stage_flow,
        f"throughput: {rate:.1f} units/s | ETA {_eta(payload.get('eta_seconds'))}",
        ("live: " + " | ".join(live)) if live else "live: -",
        f"work: {_gib(disk.get('work_bytes'))} | free: {_gib(disk.get('free_bytes'))} | "
        f"reserve: {_gib(disk.get('reserve_bytes'))}",
        f"updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
    ]
    return "\n".join(lines)


def render(payload: dict[str, Any]) -> str:
    stage = str(payload.get("stage") or "")
    detail = str(payload.get("detail") or "")
    if stage == "materialize" and any(phase in detail for phase in PHASES):
        return _render_balance(payload)
    return _render_stage(payload)


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

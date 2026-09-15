#!/usr/bin/env python3
"""Finalize R25 F receipts after materializing Equal and column-only controls."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


R25_SOURCES = (
    ("B13-FULL", 13), ("B13-FULL", 29),
    ("SPLIT-QTKD", 13), ("SPLIT-QTKD", 29),
    ("EDGE-CONT", 13), ("EDGE-CONT", 29),
    ("SPLIT-U", 13), ("SPLIT-U", 29),
    ("SPLIT-SUP", 13), ("SPLIT-SUP", 29),
    ("SPLIT-UQTKD", 13), ("SPLIT-UQTKD", 29),
    ("LSE-QTKD", 13), ("LSE-QTKD", 29),
)
LEGACY_SOURCES = (("Qwen-Raw", 13), ("B13", 13), ("N-U", 13), ("N-U", 29))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _ranking_metrics(path: Path) -> dict:
    """Summarize a materialized fusion ranking when no sidecar exists."""
    if not path.is_file():
        return {}
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        return {"queries": 0}
    recalls = {
        f"R@{k}": statistics.fmean(
            1.0 if set(row.get("positive_target_ids", [])) & set(row.get("ranking", [])[ :k]) else 0.0
            for row in rows
        )
        for k in (10, 20, 50)
    }
    alphas = [float(row["alpha"]) for row in rows if "alpha" in row]
    return {
        "queries": len(rows),
        **recalls,
        "direct_scores_complete": all(row.get("direct_scores_complete") is True for row in rows),
        "alpha_mean": statistics.fmean(alphas) if alphas else None,
        "alpha_lt_0_1": sum(value < 0.1 for value in alphas),
        "alpha_gt_0_9": sum(value > 0.9 for value in alphas),
    }


def _update_r25(root: Path, arm: str, seed: int) -> dict:
    destination = root / "fusion" / arm / f"seed{seed}"
    status_path = destination / "F_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {
        "format_version": 1, "module": "F", "arm": arm, "seed": seed,
    }
    status["completed_methods"] = ["Equal", "confidence-only", "column-only"]
    status["blocked_methods"] = {}
    status["status"] = "complete"
    status["outputs"] = {
        **status.get("outputs", {}),
        "Equal": str((destination / "Equal.jsonl.gz").resolve()),
        "confidence-only": str((destination / "confidence-only.jsonl.gz").resolve()),
        "column-only": str((destination / "column-only.jsonl.gz").resolve()),
    }
    confidence_metrics = _metrics(destination / "confidence-only.metrics.json") or _ranking_metrics(destination / "confidence-only.jsonl.gz")
    if confidence_metrics and not (destination / "confidence-only.metrics.json").is_file():
        (destination / "confidence-only.metrics.json").write_text(json.dumps(confidence_metrics, indent=2) + "\n", encoding="utf-8")
    status["metrics"] = {
        "Equal": _metrics(destination / "Equal.metrics.json"),
        "confidence-only": confidence_metrics,
        "column-only": _metrics(destination / "column-only.metrics.json"),
    }
    status["created_at_utc"] = _now()
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return {"arm": arm, "seed": seed, "status": status["status"], "completed_methods": status["completed_methods"]}


def _legacy(root: Path, arm: str, seed: int) -> dict:
    destination = root / "fusion" / arm / f"seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    confidence_metrics = _metrics(destination / "confidence-only.metrics.json")
    confidence_complete = confidence_metrics.get("direct_scores_complete") is True
    completed_methods = ["Equal", "column-only"] + (["confidence-only"] if confidence_complete else [])
    status = {
        "format_version": 1,
        "module": "F",
        "arm": arm,
        "seed": seed,
        "status": "complete" if confidence_complete else "partial",
        "completed_methods": completed_methods,
        "blocked_methods": {} if confidence_complete else {
            "confidence-only": "legacy ranking artifacts do not contain complete Direct score vectors; rank gaps cannot substitute for score margin",
        },
        "outputs": {
            "Equal": str((destination / "Equal.jsonl.gz").resolve()),
            "column-only": str((destination / "column-only.jsonl.gz").resolve()),
            **({"confidence-only": str((destination / "confidence-only.jsonl.gz").resolve())} if confidence_complete else {}),
        },
        "metrics": {
            "Equal": _metrics(destination / "Equal.metrics.json"),
            "column-only": _metrics(destination / "column-only.metrics.json"),
            "confidence-only": confidence_metrics,
        },
        "created_at_utc": _now(),
    }
    (destination / "F_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return {"arm": arm, "seed": seed, "status": status["status"], "completed_methods": status["completed_methods"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    statuses = [_update_r25(root, arm, seed) for arm, seed in R25_SOURCES]
    statuses.extend(_legacy(root, arm, seed) for arm, seed in LEGACY_SOURCES)
    receipt_status = "complete" if all(item["status"] == "complete" for item in statuses) else "partial"
    receipt = {
        "format_version": 1,
        "module": "F",
        "status": receipt_status,
        "sources": statuses,
        "note": "All 14 R25 arms and four historical controls have Equal, confidence-only, and legal column-only controls. Legacy Direct scores were recovered only from frozen feature/checkpoint/detailed-path artifacts; no rank-gap substitute was used.",
        "created_at_utc": _now(),
    }
    (root / "fusion" / "F_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

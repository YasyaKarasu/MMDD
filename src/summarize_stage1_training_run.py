#!/usr/bin/env python
"""Write the optimization-plan metric table for one Stage-1 training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metrics_row(scope: str, ranking: str, metrics: dict[str, Any]) -> str:
    channel = metrics if ranking == "fused" else metrics[ranking]
    return (
        f"| {scope} | {ranking} | {channel['recall@10']:.2%} | "
        f"{channel['recall@100']:.2%} | {channel['mrr@100']:.4f} |"
    )


def _metric_table(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "| Scope | Channel | R@10 | R@100 | MRR@100 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for channel in ("direct", "evidence", "fused"):
        lines.append(_metrics_row("overall", channel, metrics))
    for dataset, dataset_metrics in metrics.get("by_dataset", {}).items():
        for channel in ("direct", "evidence", "fused"):
            lines.append(_metrics_row(dataset, channel, dataset_metrics))
    return lines


def run(args: argparse.Namespace) -> str:
    history_path = Path(args.history)
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    retrieval_epochs = [
        record for record in payload["epochs"] if "dev_retrieval" in record
    ]
    if not retrieval_epochs:
        raise ValueError(f"{history_path}: no retrieval epochs")
    by_epoch = {int(record["epoch"]): record for record in retrieval_epochs}
    epoch_zero = by_epoch.get(0)
    if epoch_zero is None:
        raise ValueError(f"{history_path}: no epoch-0 retrieval record")
    best_epoch = int(payload["best_epoch"])
    best = by_epoch[best_epoch]
    baseline = float(epoch_zero["dev_retrieval"]["direct"]["recall@10"])
    violating_epochs = [
        epoch
        for epoch, record in sorted(by_epoch.items())
        if epoch > 0
        and float(record["dev_retrieval"]["direct"]["recall@10"]) < baseline
    ]
    lines = [
        f"# {args.label}",
        "",
        f"Artifact: `{history_path.parent}`",
        "",
        "## Epoch 0",
        "",
        *_metric_table(epoch_zero["dev_retrieval"]),
        "",
        f"## Best epoch ({best_epoch})",
        "",
        *_metric_table(best["dev_retrieval"]),
        "",
        "## Raw reference",
        "",
        *_metric_table(epoch_zero["dev_retrieval"]["raw_embedding"]),
        "",
        f"Epochs below epoch-0 direct R@10: {violating_epochs or 'none'}.",
        "",
    ]
    report = "\n".join(lines)
    output = Path(args.output) if args.output else history_path.parent / "RESULTS.md"
    output.write_text(report, encoding="utf-8")
    if args.summary:
        with Path(args.summary).open("a", encoding="utf-8") as handle:
            handle.write(report + "\n")
    print(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output")
    parser.add_argument("--summary")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

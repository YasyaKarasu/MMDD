#!/usr/bin/env python
"""Measure the exact WDC raw-retrieval saturation curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from mmdd_stage1.data import TargetExample, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import load_corpus_ids

DEPTHS = (10, 30, 50, 100, 200, 500, 1000)


def exact_positive_ranks(
    query_ids: Sequence[str],
    positive_ids: Sequence[Sequence[str]],
    table_ids: Sequence[str],
    query_embeddings: torch.Tensor,
    table_embeddings: torch.Tensor,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Rank every positive under exact raw inner-product retrieval."""

    if len(query_ids) != len(positive_ids) or len(query_ids) != len(query_embeddings):
        raise ValueError("query IDs, positives, and embeddings must align")
    if len(table_ids) != len(table_embeddings):
        raise ValueError("table IDs and embeddings must align")
    if query_embeddings.ndim != 2 or table_embeddings.ndim != 2:
        raise ValueError("embedding matrices must have shape [objects, dimensions]")
    if query_embeddings.shape[1] != table_embeddings.shape[1]:
        raise ValueError("query and table embedding dimensions must match")

    scores = (
        query_embeddings.to(device=device, dtype=torch.float32)
        @ table_embeddings.to(device=device, dtype=torch.float32).T
    ).cpu().numpy()
    table_id_array = np.asarray(table_ids, dtype=str)
    table_positions = {table_id: index for index, table_id in enumerate(table_ids)}
    rows = []
    for query_index, (query_id, positives) in enumerate(zip(query_ids, positive_ids)):
        # Match retrieval.py's deterministic secondary ordering by target ID.
        order = np.lexsort((table_id_array, -scores[query_index]))
        ranks = np.empty(len(table_ids), dtype=np.int64)
        ranks[order] = np.arange(1, len(table_ids) + 1)
        for positive_id in positives:
            position = table_positions.get(positive_id)
            rows.append(
                {
                    "query_id": query_id,
                    "positive_target_id": positive_id,
                    "rank": int(ranks[position]) if position is not None else None,
                    "score": (
                        float(scores[query_index, position])
                        if position is not None
                        else None
                    ),
                    "in_corpus": position is not None,
                }
            )
    return rows


def saturation_metrics(
    examples: Sequence[TargetExample],
    rank_rows: Sequence[dict[str, Any]],
    *,
    corpus_size: int,
    depths: Sequence[int] = DEPTHS,
) -> dict[str, Any]:
    """Aggregate per-query recall and positive-rank quantiles."""

    ranks_by_query: dict[str, dict[str, int | None]] = {
        example.query_id: {} for example in examples
    }
    for row in rank_rows:
        ranks_by_query[str(row["query_id"])][str(row["positive_target_id"])] = row[
            "rank"
        ]

    curve = []
    requested_depths: list[int | str] = [*depths, "all"]
    for requested in requested_depths:
        effective = corpus_size if requested == "all" else min(int(requested), corpus_size)
        per_query = []
        for example in examples:
            ranks = ranks_by_query[example.query_id]
            recalled = sum(
                ranks.get(positive_id) is not None
                and int(ranks[positive_id]) <= effective
                for positive_id in example.positive_target_ids
            )
            per_query.append(recalled / len(example.positive_target_ids))
        curve.append(
            {
                "requested_k": requested,
                "effective_k": effective,
                "recall": float(np.mean(per_query)),
                "per_query": per_query,
            }
        )

    present_ranks = [int(row["rank"]) for row in rank_rows if row["rank"] is not None]
    quantiles = (
        np.quantile(present_ranks, [0.5, 0.75, 0.95], method="linear")
        if present_ranks
        else [float("nan")] * 3
    )
    by_k = {str(row["requested_k"]): row["recall"] for row in curve}
    if by_k["50"] >= 0.85:
        decision = "saturated_current_pool"
    elif by_k["100"] < 0.8 or by_k["200"] < 0.8:
        decision = "unsaturated_current_pool"
    else:
        decision = "intermediate_current_pool"
    return {
        "queries": len(examples),
        "positive_targets": len(rank_rows),
        "positive_targets_in_corpus": len(present_ranks),
        "table_corpus_size": corpus_size,
        "curve": curve,
        "rank_quantiles": {
            "median": float(quantiles[0]),
            "p75": float(quantiles[1]),
            "p95": float(quantiles[2]),
        },
        "decision": decision,
        "scope": "current WDC table corpus only; not the full WebTable space",
    }


def _write_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
    rank_rows: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "positive_ranks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "positive_target_id",
                "rank",
                "score",
                "in_corpus",
            ],
        )
        writer.writeheader()
        writer.writerows(rank_rows)
    with (output_dir / "saturation_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["requested_k", "effective_k", "recall"]
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames} for row in metrics["curve"]
        )

    x = [int(row["effective_k"]) for row in metrics["curve"]]
    y = [float(row["recall"]) for row in metrics["curve"]]
    labels = [str(row["requested_k"]) for row in metrics["curve"]]
    from PIL import Image, ImageDraw

    width, height = 1200, 760
    left, top, right, bottom = 110, 80, 50, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    minimum_log = math.log10(min(x))
    maximum_log = math.log10(max(x))

    def x_pixel(value: int) -> int:
        fraction = (math.log10(value) - minimum_log) / (maximum_log - minimum_log)
        return round(left + fraction * plot_width)

    def y_pixel(value: float) -> int:
        return round(top + (1.0 - value) * plot_height)

    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        pixel = y_pixel(tick)
        draw.line((left, pixel, width - right, pixel), fill="#d8d8d8", width=1)
        draw.text((left - 52, pixel - 8), f"{tick:.0%}", fill="black")
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill="black",
        width=2,
    )
    points = [(x_pixel(x_value), y_pixel(y_value)) for x_value, y_value in zip(x, y)]
    draw.line(points, fill="#1769aa", width=4, joint="curve")
    for (x_value, y_value), label in zip(points, labels):
        draw.ellipse(
            (x_value - 6, y_value - 6, x_value + 6, y_value + 6),
            fill="#1769aa",
        )
        draw.text((x_value - 12, height - bottom + 14), label, fill="black")
    for point, recall in zip(points, y):
        draw.text((point[0] - 20, point[1] - 28), f"{recall:.1%}", fill="#1769aa")
    draw.text(
        (left, 24),
        "WDC raw recall saturation in the current table corpus",
        fill="black",
    )
    draw.text(
        (left + plot_width // 2 - 120, height - 48),
        "Retrieval depth k (log scale)",
        fill="black",
    )
    draw.text((12, top - 24), "Raw direct recall", fill="black")
    image.save(output_dir / "saturation_curve.png")

    by_k = {str(row["requested_k"]): row for row in metrics["curve"]}
    q = metrics["rank_quantiles"]
    lines = [
        "# Stage-1 r8 Task R1: WDC raw recall saturation",
        "",
        f"Decision: **{metrics['decision']}**.",
        "",
        f"This is exact brute-force inner-product retrieval over the current "
        f"{metrics['table_corpus_size']:,}-table WDC corpus for "
        f"{metrics['queries']} dev queries. It is not a saturation claim for the "
        "full WebTable space.",
        "",
        "| k | Effective k | Raw direct R@k |",
        "| ---: | ---: | ---: |",
    ]
    for row in metrics["curve"]:
        lines.append(
            f"| {row['requested_k']} | {row['effective_k']} | {row['recall']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Positive target rank distribution",
            "",
            f"Median={q['median']:.1f}, P75={q['p75']:.1f}, P95={q['p95']:.1f}. "
            f"{metrics['positive_targets_in_corpus']}/{metrics['positive_targets']} "
            "positive targets are present in the current corpus.",
            "",
            "The plan's 85% R@50 saturation gate is "
            + ("satisfied." if by_k["50"]["recall"] >= 0.85 else "not satisfied."),
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    data = (
        root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/wdc_stage1_data/target_lists.jsonl"
    )
    corpus = (
        root
        / "work/stage1_optimization_r4_20260829/taskJ_per_lake_baselines/corpora/wdc_corpus.jsonl"
    )
    features = (
        root
        / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    )
    store = FeatureStore.from_path(features, cache_size=args.feature_cache_size)
    examples = load_target_examples(data, split="dev", dataset_name="wdc2k_v2")
    table_ids = load_corpus_ids(corpus, store)["table"]
    query_embeddings = torch.stack(
        [store.embedding_features(example.query_id).embedding for example in examples]
    )
    table_embeddings = torch.stack(
        [store.embedding_features(table_id).embedding for table_id in table_ids]
    )
    rows = exact_positive_ranks(
        [example.query_id for example in examples],
        [example.positive_target_ids for example in examples],
        table_ids,
        query_embeddings,
        table_embeddings,
        device=torch.device(args.device),
    )
    metrics = saturation_metrics(examples, rows, corpus_size=len(table_ids))
    metrics.update(
        {
            "format_version": 1,
            "retrieval": "exact brute-force raw inner product",
            "data": str(data.resolve()),
            "corpus": str(corpus.resolve()),
            "features": str(features.resolve()),
            "device": args.device,
        }
    )
    _write_outputs(args.output_root / "taskR1_saturation_curve", metrics, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-cache-size", type=int, default=8_000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r8_20260831"),
    )
    values = parser.parse_args()
    if values.feature_cache_size <= 0:
        parser.error("--feature-cache-size must be positive")
    return values


if __name__ == "__main__":
    run(parse_args())

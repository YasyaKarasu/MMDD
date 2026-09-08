"""Metric definitions shared by Stage-2 training and reporting."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean
from typing import Any


def evidence_modality_bucket(modalities: Sequence[str]) -> str:
    kinds = set(modalities)
    if kinds == {"text"}:
        return "text_only"
    if kinds == {"image"}:
        return "image_only"
    if kinds == {"text", "image"}:
        return "text_image"
    return "unknown"


def macro_dataset_accuracy(metrics: dict[str, Any]) -> float:
    return mean(
        float(item["column_accuracy@1"])
        for item in metrics["by_dataset"].values()
    )

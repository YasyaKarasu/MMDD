"""Checkpoint-free recovery interfaces for the fresh selector path.

The engine consumes Stage-1 task/evidence records and a local generation model.
It deliberately has no selector-checkpoint argument: column selection is supplied
by the current run and every generation request is executed in order.  A caller
may inject the R7 image localizer; the default keeps the original image view.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .column_data import digest, write_jsonl
from .r4_recovery import SourceAwareRecoveryBackend


@dataclass(frozen=True)
class FreshRecoveryContract:
    candidate_scope: int = 30
    query_rows: int = 5
    max_evidence: int = 4
    max_new_tokens: int = 512
    selector_checkpoint: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"candidate_scope": self.candidate_scope, "query_rows": self.query_rows,
                "max_evidence": self.max_evidence, "max_new_tokens": self.max_new_tokens,
                "selector_checkpoint": None, "fresh": True}


class ImageLocalizer:
    """Localizer boundary used by :class:`CropEngine`.

    ``localize`` can be replaced with the Qwen V-V implementation.  The default
    is an auditable original-image fallback, which is valid when no ROI is found.
    """

    def __init__(self, localize: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None):
        self._localize = localize

    def localize(self, task: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
        if self._localize is not None:
            return dict(self._localize(task, evidence))
        return {"view_kind": "ORIGINAL", "evidence_id": evidence.get("asset_id"),
                "crop": None, "fallback_reason": "NO_LOCALIZER_INJECTED"}


class CropEngine:
    """Apply fresh image views before calling a generation backend."""

    def __init__(self, generator: Any, localizer: ImageLocalizer | None = None):
        self.generator = generator
        self.localizer = localizer or ImageLocalizer()

    def recover(self, arm: str, query_cells: list[dict], column_name: str,
                evidence: list[dict[str, Any]]) -> dict[str, Any]:
        prepared = []
        views = []
        for item in evidence:
            item = dict(item)
            if item.get("asset_type") == "image":
                view = self.localizer.localize({"query_cells": query_cells, "column_name": column_name}, item)
                views.append(view)
                if view.get("image") is not None:
                    item["image"] = view["image"]
            prepared.append(item)
        result = self.generator.recover(arm, query_cells, column_name, prepared)
        return {**result, "views": views, "fresh_recovery": True, "selector_checkpoint": None}


class QueryRunner:
    """Sequential fresh task runner with content-addressed raw outputs."""

    def __init__(self, engine: CropEngine, output: Path, *, contract: FreshRecoveryContract | None = None):
        self.engine = engine
        self.output = Path(output)
        self.contract = contract or FreshRecoveryContract()

    def run(self, tasks: Sequence[Mapping[str, Any]], *, arm: str = "R1") -> list[dict[str, Any]]:
        if self.contract.selector_checkpoint is not None:
            raise ValueError("fresh recovery forbids selector checkpoints")
        rows = []
        for task in tasks:
            evidence = list(task.get("evidence", ()))[:self.contract.max_evidence]
            result = self.engine.recover(arm, list(task["query_cells"]), str(task["column_name"]), evidence)
            rows.append({**task, **result, "task_id": digest({k: task[k] for k in task if k != "evidence"}),
                         "fresh": True, "selector_checkpoint": None})
        self.output.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.output / f"RAW_{arm}.jsonl", rows)
        (self.output / "FRESH_RECOVERY_RECEIPT.json").write_text(json.dumps({
            "tasks": len(rows), "arm": arm, "contract": self.contract.as_dict(),
            "selector_checkpoint_used": False,
            "raw_sha256": hashlib.sha256((self.output / f"RAW_{arm}.jsonl").read_bytes()).hexdigest(),
        }, indent=2) + "\n")
        return rows


def build_qwen_engine(model_dir: Path, *, device: str = "cuda:0",
                      max_image_pixels: int = 262144) -> CropEngine:
    """Create the fresh Qwen generator; no selector or prior checkpoint is loaded."""
    backend = SourceAwareRecoveryBackend(Path(model_dir), device=device,
                                          reader_image_max_pixels=max_image_pixels)
    return CropEngine(backend)

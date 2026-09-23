"""Typed records for the S2-R4c FAST image-view experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .r4_common import digest


@dataclass(frozen=True)
class FastUnit:
    unit_id: str
    query_id: str
    target_id: str
    column_id: int
    column_name: str
    query_row_id: int
    source_group: str
    cells: tuple[tuple[str, str], ...]
    evidence_ids: tuple[str, ...]
    focus_image_id: str

    @classmethod
    def create(
        cls,
        *,
        query_id: str,
        target_id: str,
        column_id: int,
        column_name: str,
        query_row_id: int,
        source_group: str,
        cells: tuple[tuple[str, str], ...],
        evidence_ids: tuple[str, ...],
        focus_image_id: str,
    ) -> "FastUnit":
        inference = {
            "query_id": query_id,
            "target_id": target_id,
            "column_id": int(column_id),
            "column_name": column_name,
            "query_row_id": int(query_row_id),
            "source_group": source_group,
            "cells": [list(cell) for cell in cells],
            "evidence_ids": list(evidence_ids),
            "focus_image_id": focus_image_id,
        }
        return cls(
            unit_id=digest(inference),
            query_id=query_id,
            target_id=target_id,
            column_id=int(column_id),
            column_name=column_name,
            query_row_id=int(query_row_id),
            source_group=source_group,
            cells=cells,
            evidence_ids=evidence_ids,
            focus_image_id=focus_image_id,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FastUnit":
        return cls(
            unit_id=str(value["unit_id"]),
            query_id=str(value["query_id"]),
            target_id=str(value["target_id"]),
            column_id=int(value["column_id"]),
            column_name=str(value["column_name"]),
            query_row_id=int(value["query_row_id"]),
            source_group=str(value["source_group"]),
            cells=tuple((str(name), str(text)) for name, text in value["cells"]),
            evidence_ids=tuple(str(item) for item in value["evidence_ids"]),
            focus_image_id=str(value["focus_image_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cells"] = [list(cell) for cell in self.cells]
        value["evidence_ids"] = list(self.evidence_ids)
        return value

    def row_dict(self) -> dict[str, str]:
        return dict(self.cells)


@dataclass(frozen=True)
class ViewSpec:
    source_asset_id: str
    view_id: str
    local_path: str
    pixel_box: tuple[int, int, int, int] | None
    max_pixels: int
    view_sha256: str

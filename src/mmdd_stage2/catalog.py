"""Read-only object store for one dataset: query tables, lake tables and evidence assets.

``build_catalog`` streams the dataset artifact once into ``<run>/catalog.sqlite``; later steps look
objects up by id instead of holding the 20k-table lake in memory. Labels are kept out of it
(qrels are read only by ``jobs`` for training labels and ``evaluate``).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from mmdd_dataset.wdc_runtime import iter_dataset_artifact

from .common import write_json
from .values import ROWS, query_rows, table_domain

SPLITS = ("train", "dev", "test")


def reader_table(table: dict[str, Any]) -> dict[str, Any]:
    """The visible table the selector reader serializes: headers and cell texts only."""
    return {"table_id": str(table["table_id"]),
            "columns": [{"column_index": int(c["column_index"]), "column_name": str(c["column_name"])}
                        for c in table["columns"]],
            "rows": [{"cells": [{"column_index": int(c["column_index"]),
                                 "text": "" if c.get("text") is None else str(c["text"])} for c in row["cells"]]}
                     for row in table["rows"]]}


def source_columns(table: dict[str, Any]) -> dict[str, int]:
    """Source-table column index -> local column index (qrels name gold columns by source index)."""
    return {str(int(c.get("source_column_index", c["column_index"]))): int(c["column_index"]) for c in table["columns"]}


def build_catalog(dataset_root: Path, run: Path) -> None:
    path = run / "catalog.sqlite"
    if path.exists():
        raise FileExistsError(f"{path} already built")
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    db.execute("CREATE TABLE objects(kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind, id))")

    def add(kind: str, key: str, value: Any) -> None:
        db.execute("INSERT INTO objects VALUES(?,?,?)", (kind, key, json.dumps(value, ensure_ascii=False)))

    population: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    for table in iter_dataset_artifact(dataset_root, "query_tables"):
        query_id = str(table["table_id"])
        if len(table["rows"]) != ROWS:
            raise ValueError(f"{query_id}: query tables must hold exactly {ROWS} rows")
        visible = reader_table(table)
        add("query", query_id, visible)
        add("query_rows", query_id, query_rows(visible))
        add("query_sources", query_id, source_columns(table))
        population[table["split"]].append({"query_id": query_id, "split": table["split"],
                                           "source_group": str(table["source_table_id"])})
    sources: dict[str, dict] | None = None
    for table in iter_dataset_artifact(dataset_root, "data_lake_tables"):
        if "source_table_ref" in table:
            if sources is None:
                sources = {
                    str(r["source_table_id"]): r
                    for r in iter_dataset_artifact(dataset_root, "source_tables")
                }
            sid = str(table["source_table_ref"]["source_table_id"])
            table = {**sources[sid], **{k: v for k, v in table.items() if k != "source_table_ref"}}
        target_id = str(table["table_id"])
        add("target", target_id, reader_table(table))
        add("target_sources", target_id, source_columns(table))
        add("target_domain", target_id, table_domain(table))
    for asset in iter_dataset_artifact(dataset_root, "bridge_assets"):
        add("asset", str(asset["asset_id"]),
            {"asset_id": str(asset["asset_id"]), "asset_type": asset["asset_type"],
             **({"content": asset.get("content", "")} if asset["asset_type"] == "text"
                else {"local_path": asset["local_path"]})})
    db.commit()
    db.close()
    temporary.replace(path)

    groups = {split: {row["source_group"] for row in rows} for split, rows in population.items()}
    if groups["train"] & groups["dev"] or groups["train"] & groups["test"] or groups["dev"] & groups["test"]:
        raise ValueError("a source group appears in more than one split")
    for split, rows in population.items():
        write_json(run / "population" / f"{split}.json", sorted(rows, key=lambda row: row["query_id"]))
    print(json.dumps({"catalog": str(path), "queries": {s: len(r) for s, r in population.items()}}), flush=True)


class Catalog:
    def __init__(self, run: Path) -> None:
        self.db = sqlite3.connect(f"file:{run / 'catalog.sqlite'}?mode=ro", uri=True)

    def get(self, kind: str, key: str) -> Any:
        row = self.db.execute("SELECT payload FROM objects WHERE kind=? AND id=?", (kind, key)).fetchone()
        if row is None:
            raise KeyError(f"{kind}:{key} not in catalog")
        return json.loads(row[0])

    def evidence(self, asset_id: str) -> dict[str, Any]:
        asset = self.get("asset", asset_id)
        if asset["asset_type"] == "image" and not Path(asset["local_path"]).is_file():
            raise FileNotFoundError(f"{asset_id}: missing evidence image {asset['local_path']}")
        return asset

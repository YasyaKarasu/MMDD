#!/usr/bin/env python
"""Build a standalone image-attribute extraction subset from MMDD artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage1_io import iter_manifest_records, write_json, write_jsonl


DEFAULT_OUTPUT_DIR = "output_mm_image_attribute_5k"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Existing mm_joinability dataset directory.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Standalone output directory.")
    parser.add_argument("--total_samples", type=int, default=2000)
    parser.add_argument("--train_samples", type=int, default=1600)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--positive_samples", type=int, default=1369)
    parser.add_argument("--negative_samples", type=int, default=631)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max_samples_per_table", type=int, default=20)
    parser.add_argument("--max_samples_per_entity", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    counts = (args.total_samples, args.train_samples, args.val_samples, args.test_samples,
              args.positive_samples, args.negative_samples, args.max_samples_per_table,
              args.max_samples_per_entity)
    if any(value <= 0 for value in counts):
        raise ValueError("all sample counts and caps must be positive")
    if args.train_samples + args.val_samples + args.test_samples != args.total_samples:
        raise ValueError("split sample counts must equal total_samples")
    if args.positive_samples + args.negative_samples != args.total_samples:
        raise ValueError("positive_samples + negative_samples must equal total_samples")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def normalize(value: Any) -> str:
    return clean_text(value).casefold()


def values_match(predicted: Any, expected: Any) -> bool:
    left, right = normalize(predicted), normalize(expected)
    return bool(left and right and (left == right or (len(right) >= 4 and (left in right or right in left))))


def column_bucket(num_cols: int) -> str:
    return "2-4" if num_cols <= 4 else "5-7" if num_cols <= 7 else "8-12" if num_cols <= 12 else "13+"


def public_sample(sample: dict[str, Any], image_suffix: str) -> dict[str, Any]:
    record = {key: value for key, value in sample.items() if key != "asset_id"}
    record["image_path"] = (Path("images") / f"{sample['sample_id']}{image_suffix}").as_posix()
    return record


def _cells(table: dict[str, Any], row_id: int) -> dict[str, str]:
    for row in table.get("rows", []):
        if row.get("row_id") == row_id:
            return {clean_text(cell.get("column_name")): clean_text(cell.get("text")) for cell in row.get("cells", [])}
    return {}


def build_candidates(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    valid_assets = {
        row["asset_id"]
        for row in iter_manifest_records(input_dir, "bridge_assets")
        if row.get("asset_type") == "image" and Path(row.get("local_path", "")).is_file()
    }
    excluded: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="mm_attr_candidates_") as work:
        database = sqlite3.connect(Path(work) / "candidates.sqlite")
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("CREATE TABLE extraction (table_id TEXT, payload TEXT)")
        batch: list[tuple[str, str]] = []
        for extraction in iter_manifest_records(input_dir, "attribute_extractions"):
            if extraction.get("asset_type") != "image" or extraction.get("asset_id") not in valid_assets:
                continue
            batch.append((clean_text(extraction.get("source_table_id")), json.dumps(extraction, ensure_ascii=False)))
            if len(batch) >= 5000:
                database.executemany("INSERT INTO extraction VALUES (?, ?)", batch)
                batch.clear()
        if batch:
            database.executemany("INSERT INTO extraction VALUES (?, ?)", batch)
        database.execute("CREATE INDEX extraction_table_idx ON extraction(table_id)")
        database.execute("CREATE TABLE candidate (label INTEGER, bucket TEXT, score TEXT, payload TEXT)")
        candidate_batch: list[tuple[int, str, str, str]] = []
        for table in iter_manifest_records(input_dir, "source_tables"):
            table_id = clean_text(table.get("source_table_id"))
            for (payload,) in database.execute("SELECT payload FROM extraction WHERE table_id = ?", (table_id,)):
                extraction = json.loads(payload)
                row_id = extraction.get("source_row_id")
                entity_id = extraction.get("entity_id")
                asset_id = extraction.get("asset_id")
                cells = _cells(table, row_id)
                if not cells:
                    excluded["missing_source_row"] += 1
                    continue
                names = [clean_text(name) for name in extraction.get("candidate_attribute_names", [])]
                names = [name for name in names if name in cells and cells[name]]
                extracted = {
                    normalize(item.get("name")): item
                    for item in extraction.get("attributes", [])
                    if isinstance(item, dict) and clean_text(item.get("name"))
                }
                bucket = column_bucket(int(table.get("num_cols", 0)))
                base = {
                    "source_table_id": table_id,
                    "source_row_id": row_id,
                    "entity_id": entity_id,
                    "entity_wiki_title": extraction.get("entity_wiki_title", ""),
                    "asset_id": asset_id,
                    "num_cols": table.get("num_cols", 0),
                    "column_bucket": bucket,
                }
                for name in names:
                    item = extracted.get(normalize(name))
                    label = item is not None
                    value_match = bool(item and values_match(item.get("value"), cells[name]))
                    if item is not None and not value_match:
                        excluded["positive_value_mismatch"] += 1
                    sample_id = hashlib.sha1(f"{asset_id}|{table_id}|{row_id}|{name}|{int(label)}".encode()).hexdigest()[:20]
                    record = {
                        **base,
                        "sample_id": sample_id,
                        "attribute_name": name,
                        "ground_truth_value": cells[name],
                        "extractable": label,
                        "evidence": clean_text(item.get("evidence")) if item else "",
                        "connection_evidence": clean_text(item.get("connection_evidence")) if item else "",
                    }
                    score = hashlib.sha1(f"20260720|{sample_id}".encode()).hexdigest()
                    candidate_batch.append((int(label), bucket, score, json.dumps(record, ensure_ascii=False)))
                    if len(candidate_batch) >= 5000:
                        database.executemany("INSERT INTO candidate VALUES (?, ?, ?, ?)", candidate_batch)
                        candidate_batch.clear()
        if candidate_batch:
            database.executemany("INSERT INTO candidate VALUES (?, ?, ?, ?)", candidate_batch)
        database.execute("CREATE INDEX candidate_pick_idx ON candidate(label, bucket, score)")
        positives: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        for label, target in ((1, positives), (0, negatives)):
            for bucket in ("2-4", "5-7", "8-12", "13+"):
                rows = database.execute(
                    "SELECT payload FROM candidate WHERE label = ? AND bucket = ? ORDER BY score LIMIT 20000",
                    (label, bucket),
                )
                target.extend(json.loads(payload) for (payload,) in rows)
        database.close()
    return positives, negatives, excluded


def _take(rows: list[dict[str, Any]], count: int, seed: int, table_cap: int, entity_cap: int, strict: bool = True) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    rng = random.Random(seed); rows = list(rows); rng.shuffle(rows); result=[]; tables=Counter(); entities=Counter()
    for row in rows:
        if tables[row["source_table_id"]] >= table_cap or entities[row["entity_id"]] >= entity_cap: continue
        result.append(row); tables[row["source_table_id"]] += 1; entities[row["entity_id"]] += 1
        if len(result) == count: return result
    if strict:
        raise ValueError(f"only selected {len(result)} of required {count} samples")
    return result


def _balanced_take(rows: list[dict[str, Any]], count: int, seed: int, table_cap: int, entity_cap: int) -> list[dict[str, Any]]:
    buckets = ("2-4", "5-7", "8-12", "13+")
    base, remainder = divmod(count, len(buckets))
    selected: list[dict[str, Any]] = []
    used_entities: set[str] = set()
    for index, bucket in enumerate(buckets):
        quota = base + (1 if index < remainder else 0)
        pool = [row for row in rows if row["column_bucket"] == bucket and row["entity_id"] not in used_entities]
        part = _take(pool, quota, seed + index, table_cap, entity_cap, strict=False)
        selected.extend(part); used_entities.update(row["entity_id"] for row in part)
    if len(selected) < count:
        filler = [row for row in rows if row["entity_id"] not in used_entities]
        selected.extend(_take(filler, count - len(selected), seed + 100, table_cap, entity_cap))
    return selected


def run(args: argparse.Namespace) -> None:
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    positives, negatives, excluded = build_candidates(input_dir)
    split_sizes = {"train": args.train_samples, "val": args.val_samples, "test": args.test_samples}

    def assigned_split(table_id: str) -> str:
        value = int(hashlib.sha1(f"{args.seed}|{table_id}".encode()).hexdigest(), 16) % 10
        return "train" if value < 8 else "val" if value == 8 else "test"

    def one_per_entity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chosen: dict[str, dict[str, Any]] = {}
        for row in rows:
            entity_id = row["entity_id"]
            previous = chosen.get(entity_id)
            if previous is None or row["sample_id"] < previous["sample_id"]:
                chosen[entity_id] = row
        return list(chosen.values())

    positives = one_per_entity(positives)
    if len(positives) < args.positive_samples:
        raise ValueError(f"only {len(positives)} unique positive entities for {args.positive_samples} requested")
    positives.sort(key=lambda row: hashlib.sha1(f"{args.seed}|{row['sample_id']}".encode()).hexdigest())
    positives = positives[:args.positive_samples]
    positive_entity_ids = {row["entity_id"] for row in positives}
    negatives = one_per_entity([row for row in negatives if row["entity_id"] not in positive_entity_ids])

    by_split: dict[str, list[dict[str, Any]]] = {}
    for index, (split, size) in enumerate(split_sizes.items()):
        positive_rows = [row for row in positives if assigned_split(row["source_table_id"]) == split]
        positive_count = len(positive_rows)
        negative_count = size - positive_count
        if negative_count < 0:
            raise ValueError(f"{split} has {positive_count} positives but only {size} total slots")
        negative_pool = [row for row in negatives if assigned_split(row["source_table_id"]) == split]
        items = positive_rows
        items += _balanced_take(negative_pool, negative_count, args.seed + index * 100 + 10, args.max_samples_per_table, 1)
        random.Random(args.seed + index).shuffle(items)
        by_split[split] = items
    chosen = [sample for samples in by_split.values() for sample in samples]
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite: raise ValueError(f"output exists: {output_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".", dir=output_dir.parent or Path(".")))
    selected_table_ids = {sample["source_table_id"] for sample in chosen}
    selected_asset_ids = {sample["asset_id"] for sample in chosen}
    table_map = {row["source_table_id"]: row for row in iter_manifest_records(input_dir, "source_tables") if row.get("source_table_id") in selected_table_ids}
    asset_map = {row["asset_id"]: row for row in iter_manifest_records(input_dir, "bridge_assets") if row.get("asset_id") in selected_asset_ids}
    for split, samples in by_split.items():
        public_samples=[]; tables=[]; seen_tables=set()
        for sample in samples:
            asset=asset_map[sample["asset_id"]]; source=Path(asset["local_path"]); suffix=source.suffix or ".img"; relative=Path("images") / f"{sample['sample_id']}{suffix}"
            target=temporary / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            public_samples.append(public_sample(sample, suffix))
            if sample["source_table_id"] not in seen_tables: tables.append(table_map[sample["source_table_id"]]); seen_tables.add(sample["source_table_id"])
        write_jsonl(temporary / "samples" / f"{split}.jsonl", public_samples); write_jsonl(temporary / "tables" / f"{split}.jsonl", tables)
    write_json(temporary / "stats.json", {"selected": {key: len(value) for key, value in by_split.items()}, "positives": sum(x["extractable"] for x in chosen), "negatives": sum(not x["extractable"] for x in chosen), "exclusions": dict(excluded)})
    write_json(temporary / "manifest.json", {"format": "mm_image_attribute_dataset_v1", "input_dir": str(input_dir), "seed": args.seed})
    if output_dir.exists(): shutil.rmtree(output_dir)
    temporary.rename(output_dir)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

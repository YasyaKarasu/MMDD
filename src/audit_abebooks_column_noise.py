"""Audit column discrimination and train-only retrieval overlap without changing data."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


MISSING = {"", "-", "n/a", "none", "null"}
PROPOSALS = {
    "inventory_and_seller_rating": {"availability_quantity", "seller_rating"},
    "plus_condition_and_binding": {
        "availability_quantity", "seller_rating", "copy_condition_grade", "binding"},
    "plus_edition_number": {
        "availability_quantity", "seller_rating", "copy_condition_grade", "binding",
        "edition_number"},
}


def normalize(value: object) -> str:
    """Normalize spelling only; do not silently equate categories or drop zero."""
    text = "" if value is None else str(value)
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_artifact(root: Path, name: str) -> list[dict]:
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    return [row for shard in manifest["artifacts"][name]["shards"]
            for row in read_rows(root / shard["path"])]


def distribution(counts: Counter) -> dict:
    total = sum(counts.values())
    valid = Counter({v: n for v, n in counts.items() if v not in MISSING})
    populated = sum(valid.values())
    collision = sum((n / populated) ** 2 for n in valid.values()) if populated else None
    entropy = -sum(n / populated * math.log2(n / populated) for n in valid.values())
    return {
        "cells": total, "populated": populated, "missing": total - populated,
        "missing_rate": (total - populated) / total if total else None,
        "distinct": len(valid), "collision_probability": collision,
        "entropy_bits": entropy, "effective_distinct": 2 ** entropy if populated else 0,
        "top_values": [{"value": v, "count": n, "share_populated": n / populated,
                        "share_all": n / total} for v, n in valid.most_common(10)],
    }


def cell_counts(table: dict) -> dict[str, Counter]:
    counts = {c["column_name"]: Counter() for c in table["columns"]}
    for row in table["rows"]:
        cells = {c["column_name"]: normalize(c.get("text")) for c in row["cells"]}
        for name in counts:
            counts[name][cells.get(name, "")] += 1
    return counts


def value_sets(table: dict) -> dict[str, set[str]]:
    return {name: set(counts) - MISSING for name, counts in cell_counts(table).items()}


def summarize_columns(tables: list[dict]) -> dict[str, dict]:
    counts = defaultdict(Counter)
    per_table = defaultdict(list)
    for table in tables:
        for name, values in cell_counts(table).items():
            counts[name].update(values)
            per_table[name].append(values)
    result = {}
    for name, values in sorted(counts.items()):
        row = distribution(values)
        row["tables"] = len(per_table[name])
        row["constant_nonempty_tables"] = sum(
            len(set(c) - MISSING) == 1 for c in per_table[name])
        row["all_missing_tables"] = sum(not (set(c) - MISSING) for c in per_table[name])
        for top in row["top_values"]:
            top["tables_containing_value"] = sum(top["value"] in c for c in per_table[name])
        result[name] = row
    return result


def join_impact(columns: set[str], qrels: list[dict], recoveries: list[dict]) -> dict:
    positive = [r for r in qrels if r["rel"] > 0]
    affected = [r for r in positive if r["join_attribute"]["column_name"] in columns]
    affected_ids = {r["query_table_id"] for r in affected}
    surviving_ids = {r["query_table_id"] for r in positive
                     if r["join_attribute"]["column_name"] not in columns}
    return {
        "columns": sorted(columns), "positive_pairs": len(affected),
        "queries": len(affected_ids),
        "pairs_by_split": dict(Counter(r["split"] for r in affected)),
        "pairs_by_reason": dict(Counter(r["reason"] for r in affected)),
        "queries_losing_all_gold": len(affected_ids - surviving_ids),
        "queries_losing_all_gold_by_split": dict(Counter(
            next(r["split"] for r in affected if r["query_table_id"] == qid)
            for qid in affected_ids - surviving_ids)),
        "recovery_records": sum(r["recovered_attribute"]["column_name"] in columns
                                for r in recoveries),
        "affected_query_ids": sorted(affected_ids),
    }


def retrieval_overlap(queries: list[dict], targets: list[dict], pools: list[dict],
                      qrels: list[dict]) -> dict:
    """Count same-column shared values; gold status is query-specific."""
    qvalues = {t["table_id"]: value_sets(t) for t in queries}
    tvalues = {t["table_id"]: value_sets(t) for t in targets}
    gold = defaultdict(set)
    for row in qrels:
        if row["rel"] > 0:
            gold[row["query_table_id"]].add(row["target_table_id"])
    stats = {g: defaultdict(Counter) for g in ("top10_nongold", "gold", "all_nongold")}
    totals = Counter()
    popularity = Counter()
    for pool in pools:
        assert pool["split"] == "train", "Recommendations must use train retrieval only"
        qid = pool["query_id"]
        top = [tid for tid in pool["D100_ANN"][:10] if tid not in gold[qid]]
        popularity.update(top)
        groups = {"top10_nongold": top, "gold": sorted(gold[qid]),
                  "all_nongold": [tid for tid in tvalues if tid not in gold[qid]]}
        for group, tids in groups.items():
            totals[group] += len(tids)
            for tid in tids:
                for name in qvalues[qid].keys() & tvalues[tid].keys():
                    count = stats[group][name]
                    count["both_have_column"] += 1
                    qset, tset = qvalues[qid][name], tvalues[tid][name]
                    if qset and tset:
                        count["both_have_valid_value"] += 1
                    if qset & tset:
                        count["share_any_value"] += 1
    result = {}
    for group, columns in stats.items():
        result[group] = {name: {**{key: count[key] for key in
            ("both_have_column", "both_have_valid_value", "share_any_value")},
            "overlap_rate_all_pairs": count["share_any_value"] / totals[group],
            "overlap_rate_when_populated": count["share_any_value"] /
                count["both_have_valid_value"] if count["both_have_valid_value"] else None}
            for name, count in sorted(columns.items())}
    return {"train_queries": len(pools), "pair_counts": dict(totals), "columns": result,
            "top10_nongold_target_frequency": popularity.most_common(20)}


def recovery_ambiguity(targets: list[dict], qrels: list[dict], recoveries: list[dict]) -> dict:
    """How many other target tables contain the exact annotated recovered value?"""
    index = defaultdict(set)
    for table in targets:
        for name, values in value_sets(table).items():
            for value in values:
                index[name, value].add(table["table_id"])
    gold = defaultdict(set)
    for row in qrels:
        if row["rel"] > 0:
            gold[row["query_table_id"]].add(row["target_table_id"])
    units = {}
    for row in recoveries:
        if row["split"] != "train":
            continue
        name = row["recovered_attribute"]["column_name"]
        value = normalize(row["recovered_attribute"]["value"])
        if value not in MISSING:
            key = (row["query_table_id"], row["query_row_id"], name, value)
            units[key] = len(index[name, value] - gold[row["query_table_id"]])
    grouped = defaultdict(list)
    for (_, _, name, _), count in units.items():
        grouped[name].append(count)
    return {name: {"train_unique_recovery_units": len(counts),
                   "mean_other_targets_with_exact_value": statistics.mean(counts),
                   "median_other_targets_with_exact_value": statistics.median(counts),
                   "min": min(counts), "max": max(counts)}
            for name, counts in sorted(grouped.items())}


def redundancy(tables: list[dict]) -> dict:
    triples = matches = rating_missing_equal = books = 0
    for table in tables:
        if table["source_file"] != "book":
            continue
        for row in table["rows"]:
            values = {c["column_name"]: normalize(c.get("text")) for c in row["cells"]}
            books += 1
            rating_missing_equal += ((values["goodreads_rating"] in MISSING) ==
                                     (values["goodreads_rating_count"] in MISSING))
            if all(values[c] not in MISSING for c in ("price", "shipping_price", "total_price")):
                triples += 1
                matches += math.isclose(float(values["price"]) + float(values["shipping_price"]),
                                        float(values["total_price"]), abs_tol=1e-8)
    return {"price_triples_populated": triples, "total_equals_price_plus_shipping": matches,
            "book_rows": books, "rating_and_count_missingness_equal": rating_missing_equal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    baseline = root / "baseline/dataset_view"
    source = read_artifact(baseline, "source_tables")
    qrels = read_rows(baseline / "qrels.jsonl")
    recoveries = read_artifact(baseline, "evidence_recoveries")
    profiles = summarize_columns(source)
    arms = {}
    for arm in ("baseline", "columns"):
        dataset = root / arm / "dataset_view"
        queries = read_artifact(dataset, "query_tables")
        targets = read_artifact(dataset, "data_lake_tables")
        pools = read_rows(root / arm / "training_records/raw_train/pools.jsonl.gz")
        arms[arm] = {"query_columns": summarize_columns(queries),
                     "target_columns": summarize_columns(targets),
                     "retrieval": retrieval_overlap(queries, targets, pools, qrels),
                     "recovery_ambiguity": recovery_ambiguity(targets, qrels, recoveries)}
    kept = {c["column_name"] for t in read_artifact(root / "columns/dataset_view", "source_tables")
            for c in t["columns"]}
    for name, profile in profiles.items():
        profile["retained_in_columns_view"] = name in kept
        profile["gold_impact"] = join_impact({name}, qrels, recoveries)
    proposals = {}
    projected = {name: read_artifact(root / "columns/dataset_view", name)
                 for name in ("query_tables", "data_lake_tables")}
    for name, columns in PROPOSALS.items():
        impact = join_impact(columns, qrels, recoveries)
        impact["empty_tables_after_projection"] = {
            kind: [t["table_id"] for t in tables
                   if not ({c["column_name"] for c in t["columns"]} - columns)]
            for kind, tables in projected.items()}
        proposals[name] = impact
    assets = read_artifact(baseline, "bridge_assets")
    texts = [a for a in assets if a["asset_type"] == "text"]
    usd_ids = {a["asset_id"] for a in texts if re.search(r"\busd\b", a.get("content", ""), re.I)}
    usd_overlap = {}
    for arm in arms:
        hop = read_rows(root / arm / "training_records/raw_train/first_hop.jsonl.gz")
        selected = [r for r in hop if r["modality"] == "text"]
        usd_overlap[arm] = {"text_first_hop_slots": len(selected),
            "slots_containing_usd": sum(r["evidence_id"] in usd_ids for r in selected)}
    result = {
        "dataset": str(baseline), "source_tables": len(source),
        "source_rows": sum(len(t["rows"]) for t in source),
        "source_schemas": dict(Counter(t["source_file"] for t in source)),
        "positive_pairs": sum(r["rel"] > 0 for r in qrels),
        "normalization": "NFKC, casefold, collapse whitespace; no category aliases",
        "missing_tokens": sorted(MISSING),
        "collision_definition": "sum p(value)^2 among populated cells, draws with replacement",
        "overlap_definition": "any same-column normalized nonmissing value shared by Q and T",
        "scope": "source cells counted once per source row; retrieval is train Raw ANN only",
        "columns": profiles, "arms": arms, "redundancy": redundancy(source),
        "proposed_deletion_impact": proposals,
        "evidence_usd": {"text_assets": len(texts), "assets_with_usd": len(usd_ids),
                         "train_first_hop": usd_overlap},
        "causality": "Descriptive audit only; no new embedding or recall ablation performed",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "COLUMN_NOISE_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    fields = ["column", "retained", "cells", "missing_rate", "distinct", "top_value",
              "top_share_populated", "collision", "effective_distinct", "gold_pairs",
              "gold_recoveries", "train_pairs", "dev_pairs", "test_pairs"]
    with (args.output_dir / "COLUMN_NOISE_AUDIT.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in profiles.items():
            top = row["top_values"][0] if row["top_values"] else {}
            impact = row["gold_impact"]
            writer.writerow({"column": name, "retained": row["retained_in_columns_view"],
                "cells": row["cells"], "missing_rate": row["missing_rate"],
                "distinct": row["distinct"], "top_value": top.get("value"),
                "top_share_populated": top.get("share_populated"),
                "collision": row["collision_probability"], "effective_distinct": row["effective_distinct"],
                "gold_pairs": impact["positive_pairs"], "gold_recoveries": impact["recovery_records"],
                **{f"{split}_pairs": impact["pairs_by_split"].get(split, 0)
                   for split in ("train", "dev", "test")}})
    print(json.dumps({"output": str(args.output_dir), "columns": len(profiles),
                      "proposals": proposals}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

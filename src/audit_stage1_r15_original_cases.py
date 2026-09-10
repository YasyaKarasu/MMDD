#!/usr/bin/env python
"""Compare the supplied B13 case CSV with frozen R15 cases and raw retrieval."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INTEGER_FIELDS = {"denominator", "support_rows", "text_QET_occurrences", "image_QET_occurrences"}
ID_FIELDS = {"qe_known_ids", "qet_known_ids"}
STRING_FIELDS = {"arm", "query_id", "source_table_id", "kind", "target_id"}


def dependency(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {(row["query_id"].strip(), row["target_id"].strip()): row for row in rows}
    if len(result) != len(rows) or any(not all(key) for key in result):
        raise ValueError("Case rows must have nonempty unique query-target keys")
    return result


def normalize(column: str, value: str) -> Any:
    value = value.strip()
    if column in STRING_FIELDS:
        return value
    if column in INTEGER_FIELDS:
        return int(value)
    if column in ID_FIELDS:
        return sorted(value.split("|")) if value else []
    if value.lower() not in {"true", "false"}:
        raise ValueError(f"Invalid boolean in {column}")
    return value.lower() == "true"


def expected_values(case: dict[str, Any], witness: dict[str, Any], qe: dict[str, set[str]]) -> dict[str, Any]:
    if case["arm"] != "B13_S_full_seed13" or witness["arm"] != "s_full":
        raise ValueError("The original S13 queue must be compared with frozen B13")
    known = set(case["known_witness_ids"])
    qe_known = known & (qe["text"] | qe["image"])
    result = {
        "arm": "S13", "query_id": case["query_id"], "source_table_id": case["source_table_id"],
        "kind": case["query_kind"], "target_id": case["target_id"],
        "denominator": witness["positive_denominator"], "known_witness": bool(known),
        "D100": case["in_ann_D100"], "E_pool": case["in_E"], "union": case["in_U"],
        "e_only": case["ann_evidence_only"], "e_only_vs_B13": case["ann_evidence_only"],
        "QE_known": bool(qe_known), "text_QE_known": bool(known & qe["text"]),
        "image_QE_known": bool(known & qe["image"]), "QET_known": bool(case["known_qet_witness_ids"]),
        "text_QET": "text" in case["known_qet_modalities"],
        "image_QET": "image" in case["known_qet_modalities"],
        "text_QET_occurrences": sum(path["evidence_type"] == "text" for path in case["known_qet_paths"]),
        "image_QET_occurrences": sum(path["evidence_type"] == "image" for path in case["known_qet_paths"]),
        "known_top_path": witness["raw_top_path_known"],
        "support_rows": case["raw_supported_row_count"],
        "qe_known_ids": sorted(qe_known), "qet_known_ids": sorted(case["known_qet_witness_ids"]),
    }
    for label, rule in (("F1", "f1_union_direct"), ("RRF", "union_rrf_equal")):
        rank = case["final_rank_by_rule"][rule]
        for k in (10, 20, 50):
            result[f"{label}@{k}"] = rank is not None and rank <= k
    return result


def audit(root: Path, original_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    output = root / "work/stage1_optimization_r15_20260909"
    stage_c = output / "stageC_candidate_delivery"
    original_path = original_path or root / "B13_evidence_only_known_witness_cases.csv"
    with original_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        original = list(reader)
    reconstructed_csv = stage_c / "B13_evidence_only_known_witness_cases.csv"
    with reconstructed_csv.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    queue_path = stage_c / "B13_evidence_only_known_witness_cases.jsonl.gz"
    queue = list(read_jsonl(queue_path))
    original_by_key, cases, csv_by_key = keyed(original), keyed(queue), keyed(csv_rows)
    if set(cases) != set(csv_by_key):
        raise ValueError("Existing reconstructed CSV and JSONL key sets disagree")
    witness_path = output / "witness_funnel.jsonl.gz"
    witnesses = keyed([row for row in read_jsonl(witness_path)
                       if row["arm"] == "s_full" and (row["query_id"], row["target_id"]) in cases])
    frozen = json.loads((stage_c / "C_CONFIG_FROZEN.json").read_text(encoding="utf-8"))
    pool_path = Path(frozen["inputs"]["pool"]["path"])
    if dependency(pool_path)["sha256"] != frozen["inputs"]["pool"]["sha256"]:
        raise ValueError("Frozen B13 raw retrieval file changed")
    query_ids = {query_id for query_id, _ in cases}
    qe_by_query = {}
    for row in read_jsonl(pool_path):
        if row["query_id"] not in query_ids:
            continue
        qe = {"text": set(), "image": set()}
        for paths in row["paths_by_target"].values():
            for path in paths:
                if path["kind"] == "evidence":
                    qe[path["evidence_type"]].add(path["evidence_id"])
        qe_by_query[row["query_id"]] = qe
    if set(qe_by_query) != query_ids:
        raise ValueError("Frozen raw pool does not cover every queued query")
    missing = sorted(set(original_by_key) - set(cases))
    extra = sorted(set(cases) - set(original_by_key))
    fields = {name: {"compared": 0, "mismatches": 0} for name in columns}
    mismatches = []
    row_checks = []
    for key in sorted(set(original_by_key) & set(cases)):
        expected = expected_values(cases[key], witnesses[key], qe_by_query[key[0]])
        if set(expected) != set(columns):
            raise ValueError("Original CSV schema differs from the 30 defined field comparisons")
        failed = []
        for name in columns:
            actual = normalize(name, original_by_key[key][name])
            fields[name]["compared"] += 1
            if actual != expected[name]:
                failed.append(name)
                fields[name]["mismatches"] += 1
                mismatches.append({"query_id": key[0], "target_id": key[1], "column": name,
                                   "original": actual, "reconstructed": expected[name]})
        row_checks.append({"query_id": key[0], "target_id": key[1], "mismatched_fields": failed})
    normalized = [{name: normalize(name, row[name]) for name in columns} for row in original]
    identity_verified = not missing and not extra and len(original_by_key) == 82
    result = {
        "status": "passed" if identity_verified and not mismatches else "differences_found",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "identity_verified": identity_verified,
        "original": dependency(original_path), "reconstructed_csv": dependency(reconstructed_csv),
        "reconstructed_queue": dependency(queue_path), "witness_funnel": dependency(witness_path),
        "raw_B13_pool": dependency(pool_path),
        "profile": {
            "rows": len(original), "columns": len(columns), "column_names": columns,
            "unique_pairs": len(original_by_key), "queries": len({key[0] for key in original_by_key}),
            "sources": len({row["source_table_id"] for row in original}),
            "missing_keys": missing, "extra_keys": extra,
            "empty_cells": {name: sum(not row[name].strip() for row in original) for name in columns},
            "row_order_equal": list(original_by_key) == list(cases),
            "text_QET_pairs": sum(row["text_QET"] for row in normalized),
            "image_QET_pairs": sum(row["image_QET"] for row in normalized),
            "both_modalities_pairs": sum(row["text_QET"] and row["image_QET"] for row in normalized),
            "support_row_histogram": dict(sorted(Counter(row["support_rows"] for row in normalized).items())),
            "delivered_pairs": {f"{rule}@{k}": sum(row[f"{rule}@{k}"] for row in normalized)
                                for rule in ("F1", "RRF") for k in (10, 20, 50)},
        },
        "original_column_count": len(columns), "fields": fields, "mismatches": mismatches,
        "semantic_cells_checked": sum(value["compared"] for value in fields.values()),
        "semantic_cells_matched": sum(value["compared"] - value["mismatches"] for value in fields.values()),
        "row_checks": row_checks,
        "normalization": "Trim surrounding whitespace; parse integer/boolean fields; sort pipe-delimited ID lists; compare S13 to frozen B13/S-full13; ignore CSV row/column order.",
        "QE_source": "Known witness IDs intersect the union of QE evidence IDs in the frozen untruncated raw path pool, separately by modality.",
        "temporal_scope": "Static historical queue supplied on 2026-09-10, not a time-series or a newly selected evaluation cohort.",
        "original_file_modified": False, "new_training_or_model_calls": 0,
        "independent_value_or_join_verification": None,
        "scope_limit": "Verifies the user-supplied CSV against saved retrieval facts, not independent attribute/value/entity/join truth or external authorship.",
        "code": dependency(Path(__file__)),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.root, arguments.original)
    destination = arguments.root / "work/stage1_optimization_r15_20260909/ORIGINAL_CASES_VALIDATION.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "identity_verified", "semantic_cells_checked", "semantic_cells_matched", "mismatches")}))

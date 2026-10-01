#!/usr/bin/env python
"""Validate a standalone copy and export split-specific Stage-1 inputs offline."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from mmdd_dataset.abebooks_ablation import read_rows, write_rows
from mmdd_dataset.abebooks_curation import cell_values, dataset_hashes, file_hash, load_artifacts
from mmdd_dataset.abebooks_standalone import validate_queries


def validate_delivery(root: Path, stage: Path, snapshot: Path | None = None) -> dict:
    """Check row-level supervision, full labels, readers, and unchanged inputs."""
    manifest, data = load_artifacts(root)
    build = json.loads((root / "BUILD.json").read_text())
    queries = {q["table_id"]: q for q in data["query_tables"]}
    targets = {t["table_id"]: t for t in data["data_lake_tables"]}
    sources = {(s["source_table_id"], r["row_id"]): r for s in data["source_tables"] for r in s["rows"]}
    recoveries = data["evidence_recoveries"]
    qrels = read_rows(root / "qrels.jsonl")
    judgments = read_rows(root / "audit/candidate_judgments.jsonl")
    validation = validate_queries(list(queries.values()), list(targets.values()), judgments, recoveries, sources)
    assert json.loads(json.dumps(validation)) == build["validation"]
    assert manifest["curation"]["evidence_supervision"] == "recovery_records_only_no_provenance_fallback"

    catalog = json.loads((root / "retrieval_catalog.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    assert set(queries) == set(catalog["query_ids"])
    assert set(targets) == set(catalog["target_ids"])
    assert catalog["query_splits"] == splits["query_splits"] == {qid: q["split"] for qid, q in queries.items()}
    assert catalog["query_kinds"] == {qid: q["query_kind"] for qid, q in queries.items()}
    assert catalog["qrels"] == {qid: q["target_table_ids"] for qid, q in queries.items()}
    assert {(r["query_table_id"], r["target_table_id"]) for r in qrels} == {
        (qid, tid) for qid, q in queries.items() for tid in q["target_table_ids"]}
    assert len(qrels) == len({(r["query_table_id"], r["target_table_id"]) for r in qrels})

    approved = defaultdict(lambda: defaultdict(set))
    for r in recoveries:
        approved[r["query_table_id"], r["target_table_id"]][r["evidence"]["asset_id"]].add(r["query_row_id"])
    objects = {r["object_id"]: r for r in read_rows(stage / "stage1_objects.jsonl")}
    records = read_rows(stage / "target_lists.jsonl")
    assert {r["query_id"] for r in records} == set(queries)
    for record in records:
        qid = record["query_id"]
        query = queries[qid]
        size = len(query["rows"])
        assert size == record["query_row_count"] == build["query_shape"][query["query_kind"] + "_rows"]
        assert sum(p.startswith("Row: ") for p in objects[qid]["table_parts"]) == size
        if "query_context" in build:
            assert len(query["columns"]) >= 2
            assert objects[qid]["table_parts"][0] == "Columns: " + " | ".join(
                c["column_name"] for c in query["columns"])
        assert set(record["positive_target_ids"]) == set(query["target_table_ids"])
        assert record["has_recovery_supervision"] == (query["query_kind"] == "implicit")
        for tid in query["target_table_ids"]:
            expected = {aid: sorted(rows) for aid, rows in approved[qid, tid].items()}
            assert record["positive_evidence_rows_by_target"].get(tid, {}) == expected
            assert set(record["positive_evidence_by_target"].get(tid, [])) == set(expected)
            if query["query_kind"] == "implicit":
                covered = {rid for rows in expected.values() for rid in rows}
                attr = query["hidden_attributes"][0]
                assert len(covered) >= build["query_shape"]["minimum_recovered_rows"]
                assert attr["unreviewed_rows"] == size - len(covered)
                assert attr["recovered_value_ratio"] == len(covered) / size
                assert set(query["construction"]["verified_source_row_ids"]) == {
                    row["source_row_id"] for row in query["rows"] if row["row_id"] in covered}
    approved_edges = {(qid, aid) for (qid, _), evidence in approved.items() for aid in evidence}
    for edge in read_rows(stage / "edge_lists.jsonl"):
        if edge["source_type"] == "table" and edge["destination_type"] != "table":
            assert (edge["query_id"], edge["positive_id"]) in approved_edges

    summary = {"queries": len(queries), "query_shape": build["query_shape"],
               "query_row_counts": {kind: dict(Counter(len(q["rows"]) for q in queries.values()
                                                       if q["query_kind"] == kind))
                                    for kind in ("implicit", "explicit")},
               "split_counts": build["split_counts"], "validation": validation, "stage1_split_counts": {}}
    for name in ("edge_lists", "target_lists", "target_lists.evidence_supervised"):
        records = read_rows(stage / f"{name}.jsonl")
        for split in ("train", "dev", "test"):
            write_rows(stage / f"{name}.{split}.jsonl", [r for r in records if r["split"] == split])
    for split in ("train", "dev", "test"):
        examples = read_rows(stage / f"target_lists.{split}.jsonl")
        paths = read_rows(stage / f"target_lists.evidence_supervised.{split}.jsonl")
        edges = read_rows(stage / f"edge_lists.{split}.jsonl")
        expected = {qid for qid, q in queries.items() if q["split"] == split}
        assert {e["query_id"] for e in examples} == expected
        assert {e["query_id"] for e in paths} == {qid for qid in expected if queries[qid]["query_kind"] == "implicit"}
        assert all(e["query_row_count"] == len(queries[e["query_id"]]["rows"]) for e in examples)
        counts = Counter(e["query_kind"] for e in examples)
        assert counts == build["split_counts"][split]
        if build["balanced_each_split"]:
            assert counts["implicit"] == counts["explicit"] > 0
        for name, full in (("queries", list(queries.values())), ("qrels", qrels), ("recoveries", recoveries)):
            assert read_rows(root / f"splits/{split}.{name}.jsonl") == [r for r in full if r["split"] == split]
        summary["stage1_split_counts"][split] = {
            "queries": len(examples), "evidence_supervised_queries": len(paths), "edges": len(edges),
            "edge_relations": dict(Counter(f"{e['source_type']}->{e['destination_type']}" for e in edges))}

    explicit = {qid for qid, q in queries.items() if q["query_kind"] == "explicit"}
    assert {q["table_id"] for q in read_rows(root / "explicit/queries.jsonl")} == explicit
    assert read_rows(root / "explicit/qrels.jsonl") == [r for r in qrels if r["query_table_id"] in explicit]
    assert read_rows(root / "explicit/targets.jsonl") == list(targets.values())
    _, original = load_artifacts(Path(build["source"]))
    old_targets = {t["table_id"]: t for t in original["data_lake_tables"]}
    for target in targets.values():
        old = old_targets[target["provenance"]["original_table_id"]]
        assert [r["source_row_id"] for r in target["rows"]] == [r["source_row_id"] for r in old["rows"]]
        for new_row, old_row in zip(target["rows"], old["rows"]):
            old_values = cell_values(old_row)
            assert all(value == old_values[name] for name, value in cell_values(new_row).items() if name in old_values)
    assert dataset_hashes(Path(build["source"])) == build["input_hashes"]
    for name, spec in manifest["artifacts"].items():
        if name not in {"query_tables", "data_lake_tables", "evidence_recoveries"}:
            assert all(file_hash(root / shard["path"]) == build["input_hashes"][shard["path"]] for shard in spec["shards"])
    if snapshot:
        before = json.loads(snapshot.read_text())
        for name, hashes in before.items():
            assert dataset_hashes(root.parent / name) == hashes
        summary["unchanged_versions"] = list(before)
    if "query_context" in build:
        from mmdd_dataset.abebooks_query_context import direct_context_targets
        context = build["query_context"]
        parent = Path(context["source"])
        assert dataset_hashes(parent) == context["input_hashes"]
        _, parent_data = load_artifacts(parent)
        old_queries = {q["table_id"]: q for q in parent_data["query_tables"]}
        changes = read_rows(root / "audit/query_context_changes.jsonl")
        changed = {c["query_table_id"]: c for c in changes}
        for qid, query in queries.items():
            change = changed.get(qid)
            old = old_queries[change["previous_query_id"] if change else qid]
            for key in ("source_row_indices", "split", "split_group", "query_kind", "join_column", "target_table_ids"):
                assert query[key] == old[key]
            if change:
                assert not direct_context_targets(query, change["added_column"], list(targets.values()), sources)
        assert data["data_lake_tables"] == parent_data["data_lake_tables"]
        summary["query_context"] = {"changed_queries": len(changes), "minimum_visible_columns": 2,
                                    "direct_context_equality_shortcuts": 0,
                                    "rows_splits_join_columns_and_targets_preserved": True,
                                    "parent_copy_unchanged": True}
    summary["checks"] = {
        "query_sizes_and_recovery_floor_match_configuration": True,
        "stage1_serializes_all_query_rows": True,
        "stage1_supervises_only_verified_rows": True,
        "full_qrels_and_split_exports_match": True,
        "all_stage1_splits_load": True,
        "source_rows_not_reused_or_split_across_partitions": True,
        "target_membership_and_retained_cells_preserved": True,
        "source_dataset_and_preserved_artifacts_unchanged": True,
    }
    (root / "audit/DELIVERY_VALIDATION.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--input-snapshot", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_delivery(args.dataset_root, args.stage1_dir, args.input_snapshot), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

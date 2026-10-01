"""Rebuild AbeBooks views from source fields without changing task membership.

Only existing annotations are used. Every query, split, target row membership,
and positive pair must survive; inadmissible projections fail before copying.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from .abebooks_ablation import read_rows, write_rows
from .abebooks_curation import dataset_hashes, load_artifacts
from .abebooks_standalone import author_names, judge_join, project_rows, validate_queries
from .abebooks_source_rebuild import add_book_title_context, natural_author_names
from .tables import column_profiles


def normalize_qrels(records: list[dict]) -> list[dict]:
    """Use the existing Stage-1 explicit-label vocabulary, preserving all facts."""
    result = copy.deepcopy(records)
    for row in result:
        if row["reason"] == "explicit_join_column":
            row["reason"] = "explicit_visible_join_column"
            if row.get("join_attribute", {}).get("role") == "explicit_join_column":
                row["join_attribute"]["role"] = row["reason"]
    return result


def natural_author_value(value: str) -> str:
    """Invert catalog names only when the complete existing author key survives."""
    candidate = natural_author_names(value)
    names = author_names(value)
    return candidate if names and names == author_names(candidate) else value


def refresh_source_profiles(table: dict) -> None:
    """Refresh source metadata while retaining annotation-facing column indices."""
    indices = [c["column_index"] for c in table["columns"]]
    compact_columns = [{**c, "column_index": i} for i, c in enumerate(table["columns"])]
    profiles, entities = column_profiles(table["rows"], compact_columns, wiki_threshold=0.5)
    for profile in profiles:
        profile["column_index"] = indices[profile["column_index"]]
    table.update(num_rows=len(table["rows"]), num_cols=len(indices))
    table.setdefault("metadata", {}).update(column_profiles=profiles,
        candidate_entity_columns=[indices[i] for i in entities])


def project_source_dataset(source: Path, destination: Path, *,
                           drop_columns: set[str] | None = None,
                           removed_assets: set[str] | None = None,
                           natural_authors: bool = False,
                           contextualize_book_text: bool = False) -> dict:
    """Create a backed-up copy, reproject its tables, and replay all judgments."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Output must be outside the source dataset")
    if destination.exists():
        raise FileExistsError(destination)
    drop_columns, removed_assets = drop_columns or set(), removed_assets or set()
    before = dataset_hashes(source)
    manifest, data = load_artifacts(source)
    original_qrels = read_rows(source / "qrels.jsonl")
    qrels = normalize_qrels(original_qrels)
    queries = data["query_tables"]
    targets = data["data_lake_tables"]
    recoveries = data["evidence_recoveries"]
    protected_columns = {q.get("join_column", "authors") for q in queries}
    protected_columns.update(c["column_name"] for q in queries for c in q["columns"])
    if drop_columns & protected_columns:
        raise ValueError("Cannot remove a query context or join field")
    gold_assets = {r["evidence"]["asset_id"] for r in recoveries}
    asset_ids = {a["asset_id"] for a in data["bridge_assets"]}
    if removed_assets & gold_assets or not removed_assets <= asset_ids:
        raise ValueError("Removed assets must exist and cannot include annotated witnesses")

    sources = {s["source_table_id"]: s for s in data["source_tables"]}
    author_changes = []
    for table in sources.values():
        # Preserve source indices because existing facts refer to these indices.
        table["columns"] = [c for c in table["columns"] if c["column_name"] not in drop_columns]
        for row in table["rows"]:
            row["cells"] = [c for c in row["cells"] if c["column_name"] not in drop_columns]
            if natural_authors:
                for cell in row["cells"]:
                    if cell["column_name"] != "authors":
                        continue
                    before_value = str(cell.get("text") or "")
                    after_value = natural_author_value(before_value)
                    if after_value != before_value:
                        author_changes.append({"source_table_id": table["source_table_id"],
                                               "source_row_id": row["row_id"],
                                               "before": before_value, "after": after_value})
                        cell.update(text=after_value, raw=after_value)
        refresh_source_profiles(table)
    if natural_authors:
        for query in queries:
            query["columns"], query["rows"] = project_rows(sources[query["source_table_id"]],
                [r["source_row_id"] for r in query["rows"]], [c["column_name"] for c in query["columns"]])
    changes = []
    for table in targets:
        names = [c["column_name"] for c in table["columns"]]
        kept = [name for name in names if name not in drop_columns]
        if kept == names and not natural_authors:
            continue
        if len(kept) < 2 and len(kept) != len(names):
            raise ValueError("Projection would leave a target with fewer than two columns")
        original_rows = [r["source_row_id"] for r in table["rows"]]
        table["columns"], table["rows"] = project_rows(sources[table["source_table_id"]], original_rows, kept)
        table["source_column_indices"] = [c["source_column_index"] for c in table["columns"]]
        table["target_context_col_names"] = [n for n in kept if n != table.get("join_col_name")]
        changes.append({"table_id": table["table_id"], "before": names, "after": kept})
    source_rows = {(sid, row["row_id"]): row for sid, table in sources.items() for row in table["rows"]}
    judgments = [judge_join(q, t, source_rows) for q in queries for t in targets]
    original_status = {(r["query_table_id"], r["target_table_id"]): r["status"]
                       for r in read_rows(source / "audit/candidate_judgments.jsonl")}
    if any(r["status"] != original_status[r["query_table_id"], r["target_table_id"]] for r in judgments):
        raise ValueError("Source projection changes an existing query-target judgment")
    validation = validate_queries(queries, targets, judgments, recoveries, source_rows)
    counts = {split: dict(Counter(q["query_kind"] for q in queries if q["split"] == split))
              for split in ("train", "dev", "test")}
    assert all(c["implicit"] == c["explicit"] for c in counts.values())
    assert all(len(q["rows"]) == 5 and len(q["columns"]) >= 2 for q in queries)
    data["bridge_assets"] = [a for a in data["bridge_assets"] if a["asset_id"] not in removed_assets]
    for link in data["table_asset_links"]:
        link["asset_ids"] = [aid for aid in link["asset_ids"] if aid not in removed_assets]
    data["table_asset_links"] = [link for link in data["table_asset_links"] if link["asset_ids"]]
    data["attribute_extractions"] = [r for r in data["attribute_extractions"]
                                     if r["asset_id"] not in removed_assets]
    text_changes = (add_book_title_context(list(sources.values()), data["bridge_assets"])
                    if contextualize_book_text else [])
    text_by_id = {a["asset_id"]: a for a in data["bridge_assets"] if a["asset_type"] == "text"}
    transformations = {}
    for change in text_changes:
        change["content_sha256"] = hashlib.sha256(
            text_by_id[change["asset_id"]]["content"].encode()).hexdigest()
        transformations[change["asset_id"]] = {
            "operation": "prepend_existing_source_book_title",
            "original_content_sha256": change["original_content_sha256"],
            "content_sha256": change["content_sha256"],
            "original_body_preserved": True, "new_extraction_performed": False,
        }
    for record in recoveries:
        transform = transformations.get(record["evidence"]["asset_id"])
        if transform:
            assert record["evidence"]["content_sha256"] == transform["original_content_sha256"]
            record["evidence"].update(content_sha256=transform["content_sha256"],
                                      content_transformation=transform)
    for record in data["attribute_extractions"]:
        if record["asset_id"] in transformations:
            record["historical_input_transformation"] = transformations[record["asset_id"]]

    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".env.openai"))
    changed_files = ["dataset_manifest.json", "qrels.jsonl", "BUILD.json", "VALIDATION.json",
                     "retrieval_catalog.json", "REPORT.md", "TRAINING_PROTOCOL.json",
                     "audit/candidate_judgments.jsonl", "explicit/targets.jsonl", "explicit/qrels.jsonl",
                     "explicit/queries.jsonl",
                     *[f"splits/{split}.{kind}.jsonl" for split in counts for kind in ("qrels", "queries")]]
    changed_artifacts = ["source_tables", "query_tables", "data_lake_tables", "bridge_assets", "table_asset_links",
                         "attribute_extractions"]
    if text_changes:
        changed_artifacts.append("evidence_recoveries")
    changed_files.extend(shard["path"] for name in changed_artifacts
                         for shard in manifest["artifacts"][name]["shards"])
    for name in changed_files:
        backup = destination / "provenance/before_source_projection" / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination / name, backup)
    for name in changed_artifacts:
        relative = f"{name}/part-00000.jsonl"
        write_rows(destination / relative, data[name])
        manifest["artifacts"][name].update(total_records=len(data[name]),
            shards=[{"path": relative, "records": len(data[name])}])
    write_rows(destination / "qrels.jsonl", qrels)
    write_rows(destination / "audit/candidate_judgments.jsonl", judgments)
    write_rows(destination / "explicit/targets.jsonl", targets)
    explicit = {q["table_id"] for q in queries if q["query_kind"] == "explicit"}
    write_rows(destination / "explicit/queries.jsonl", [q for q in queries if q["table_id"] in explicit])
    write_rows(destination / "explicit/qrels.jsonl", [r for r in qrels if r["query_table_id"] in explicit])
    for split in counts:
        write_rows(destination / f"splits/{split}.qrels.jsonl", [r for r in qrels if r["split"] == split])
        write_rows(destination / f"splits/{split}.queries.jsonl", [q for q in queries if q["split"] == split])
    manifest["curation"].update(cache_namespace=destination.name, source_projection=True)
    catalog = json.loads((source / "retrieval_catalog.json").read_text())
    catalog["cache_namespace"] = destination.name
    protocol = json.loads((source / "TRAINING_PROTOCOL.json").read_text())
    protocol.update(dataset=str(destination), source_projection_report="SOURCE_PROJECTION.json")
    report = {"source": str(source), "destination": str(destination), "input_hashes": before,
              "dropped_source_columns": sorted(drop_columns), "removed_asset_ids": sorted(removed_assets),
              "natural_author_values": natural_authors, "author_changes": author_changes,
              "contextualize_book_text": contextualize_book_text, "text_changes": text_changes,
              "changed_targets": changes, "split_counts": counts, "queries": len(queries),
              "targets": len(targets), "assets": len(data["bridge_assets"]), "new_annotations": 0,
              "normalized_qrels": sum(a != b for a, b in zip(original_qrels, qrels)),
              "all_pair_judgments_preserved": True, "validation": validation}
    build = json.loads((source / "BUILD.json").read_text())
    build.update(destination=str(destination), validation=validation, source_projection=report,
                 assets=len(data["bridge_assets"]))
    for name, value in [("dataset_manifest.json", manifest), ("SOURCE_PROJECTION.json", report),
                        ("BUILD.json", build), ("VALIDATION.json", validation),
                        ("retrieval_catalog.json", catalog), ("TRAINING_PROTOCOL.json", protocol)]:
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (destination / "REPORT.md").write_text(
        f"# AbeBooks source projection\n\nParent: `{source}`.\n\n"
        f"Queries: {len(queries)}; targets: {len(targets)}; assets: {len(data['bridge_assets'])}. "
        "All query memberships, splits and pair judgments are preserved.\n\n"
        f"Dropped source fields: {sorted(drop_columns)}. "
        f"Normalized author fields: {len(author_changes)}. Removed assets: {len(removed_assets)}.\n\n"
        f"Text fragments with an existing source-book heading prepended: {len(text_changes)}; "
        "original bodies and existing recovery facts are preserved.\n\n"
        "No new annotations. Original files remain in the parent and modified files are backed up "
        "under `provenance/before_source_projection/`. See `SOURCE_PROJECTION.json` for checks. "
        "Retrieval performance must be assessed in a separate fresh experiment.\n")
    assert dataset_hashes(source) == before, "Original dataset changed"
    return report

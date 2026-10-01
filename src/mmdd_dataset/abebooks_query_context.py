"""Add real bibliographic context without resampling AbeBooks queries."""
from __future__ import annotations

import copy
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from .abebooks_ablation import read_rows, write_rows
from .abebooks_curation import cell_values, dataset_hashes, load_artifacts
from .abebooks_publisher import publisher_key
from .abebooks_standalone import (
    judge_join, project_rows, stable_id, text_key, validate_queries, visible_join_hint,
)


POLICY = "abebooks_two_visible_columns_context_v1"
CONTEXT_COLUMNS = ("publication_year", "publisher")


def context_key(value: str, column: str) -> str:
    if column == "publisher":
        return publisher_key(value)
    return value.strip() if re.fullmatch(r"[12]\d{3}", value.strip()) else ""


def direct_context_targets(query: dict, column: str, targets: list[dict], source_rows: dict) -> list[str]:
    """Find useful two-row joins using only the proposed visible context column.

    All returned pairs must concern the same book. Duplicate-title records are
    treated as potential matches here so uncertain identity cannot certify that
    a shortcut is absent. This is an equality-join check, not a retrieval test.
    """
    result = []
    visible = {c["column_name"] for c in query["columns"]}
    for target in targets:
        added = {c["column_name"] for c in target["columns"]} - visible
        lookup = {}
        for row in target["rows"]:
            values = cell_values(row)
            key = context_key(values.get(column, ""), column)
            if key:
                lookup.setdefault(key, []).append((row, values))
        useful, wrong = set(), False
        for row in query["rows"]:
            loc = query["source_table_id"], row["source_row_id"]
            key = context_key(cell_values(row).get(column, ""), column)
            for other, values in lookup.get(key, []):
                other_loc = target["source_table_id"], other["source_row_id"]
                same_title = text_key(cell_values(source_rows[loc])["title"]) == text_key(
                    cell_values(source_rows[other_loc])["title"])
                if loc != other_loc and not same_title:
                    wrong = True
                elif any(values.get(c) for c in added):
                    useful.add(row["row_id"])
        if len(useful) >= 2 and not wrong:
            result.append(target["table_id"])
    return result


def add_query_context(query: dict, source: dict, targets: list[dict], source_rows: dict) -> tuple[dict, list[dict]]:
    """Add one populated column, preserving hidden values and useful qrels."""
    if len(query["columns"]) >= 2:
        return copy.deepcopy(query), []
    attempts = []
    hidden = query["join_column"]
    for column in CONTEXT_COLUMNS:
        if column == hidden:
            continue
        candidate = copy.deepcopy(query)
        names = [c["column_name"] for c in query["columns"]] + [column]
        candidate["columns"], candidate["rows"] = project_rows(source, query["source_row_indices"], names)
        populated = sum(bool(context_key(cell_values(r).get(column, ""), column)) for r in candidate["rows"])
        visible = " ".join(c.get("text", "") for r in candidate["rows"] for c in r["cells"])
        reason = "accepted"
        shortcuts = []
        if populated < 3:
            reason = "insufficient_populated_context_rows"
        elif any(visible_join_hint(visible, cell_values(source_rows[
                query["source_table_id"], r["source_row_id"]])[hidden], hidden) for r in query["rows"]):
            reason = "visible_context_leaks_hidden_join_value"
        elif any(judge_join(candidate, t, source_rows)["status"] != "positive"
                 for t in targets if t["table_id"] in query["target_table_ids"]):
            reason = "context_removes_target_enrichment"
        else:
            shortcuts = direct_context_targets(candidate, column, targets, source_rows)
            if shortcuts:
                reason = "visible_context_already_enables_useful_join"
        attempts.append({"column": column, "populated_rows": populated, "reason": reason,
                         "direct_target_ids": shortcuts})
        if reason == "accepted":
            candidate["source_column_indices"] = [c["source_column_index"] for c in candidate["columns"]]
            candidate["query_context_col_names"] = names[1:]
            candidate["construction"]["visible_context_policy"] = POLICY
            return candidate, attempts
    raise ValueError(f"No safe populated context for {query['table_id']}: {attempts}")


def enrich_query_context(source: Path, destination: Path) -> dict:
    """Write a copy with new IDs only for changed query inputs; keep the lake."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be outside the source dataset")
    if destination.exists():
        raise FileExistsError(destination)
    before = dataset_hashes(source)
    manifest, data = load_artifacts(source)
    sources = {s["source_table_id"]: s for s in data["source_tables"]}
    source_rows = {(sid, r["row_id"]): r for sid, s in sources.items() for r in s["rows"]}
    targets = data["data_lake_tables"]
    queries, changes, mapping = [], [], {}
    for old in data["query_tables"]:
        query, attempts = add_query_context(old, sources[old["source_table_id"]], targets, source_rows)
        if attempts:
            qid = stable_id("query", destination.name, old["table_id"], query["source_column_indices"])
            mapping[old["table_id"]] = qid
            query.update(table_id=qid, object_id=qid)
            query["provenance"].update(context_policy=POLICY, previous_query_id=old["table_id"])
            changes.append({"previous_query_id": old["table_id"], "query_table_id": qid,
                            "source_table_id": old["source_table_id"], "split": old["split"],
                            "added_column": attempts[-1]["column"], "attempts": attempts})
        queries.append(query)
    qrels = read_rows(source / "qrels.jsonl")
    recoveries = copy.deepcopy(data["evidence_recoveries"])
    for record in qrels + recoveries:
        record["query_table_id"] = mapping.get(record["query_table_id"], record["query_table_id"])
    for r in recoveries:
        qid, tid, aid = r["query_table_id"], r["target_table_id"], r["evidence"]["asset_id"]
        r["recovery_id"] = stable_id("evrec", qid, tid, r["query_row_id"], aid)
        r["path_id"] = stable_id("path", qid, aid, tid)
        for node in r["path_nodes"]:
            node["node_id"] = mapping.get(node["node_id"], node["node_id"])
    judgments = [judge_join(q, t, source_rows) for q in queries for t in targets]
    old_status = {(mapping.get(j["query_table_id"], j["query_table_id"]), j["target_table_id"]): j["status"]
                  for j in read_rows(source / "audit/candidate_judgments.jsonl")}
    assert all(j["status"] == old_status[j["query_table_id"], j["target_table_id"]] for j in judgments)
    validation = validate_queries(queries, targets, judgments, recoveries, source_rows)
    assert all(len(q["columns"]) >= 2 for q in queries)

    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".env.openai"))
    archive = destination / "provenance/before_query_context"
    changed = ("query_tables", "evidence_recoveries", "qrels.jsonl", "splits", "splits.json",
               "retrieval_catalog.json", "explicit", "dataset_manifest.json", "BUILD.json", "REPORT.md",
               "VALIDATION.json", "TRAINING_PROTOCOL.json", "table_queryability_decisions.jsonl",
               "audit/candidate_judgments.jsonl", "audit/DELIVERY_VALIDATION.json",
               "audit/JOIN_COLUMN_VALIDATION.json", "audit/query_kind_assignments.jsonl")
    for name in changed:
        old = destination / name
        if old.exists():
            new = archive / name
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), new)
    def save_json(name, value):
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    for name, records in (("query_tables", queries), ("evidence_recoveries", recoveries)):
        path = f"{name}/part-00000.jsonl"
        write_rows(destination / path, records)
        manifest["artifacts"][name].update(total_records=len(records), shards=[{"path": path, "records": len(records)}])
    write_rows(destination / "qrels.jsonl", qrels)
    write_rows(destination / "audit/candidate_judgments.jsonl", judgments)
    write_rows(destination / "audit/query_context_changes.jsonl", changes)
    decisions = [{"query_table_id": q["table_id"], "source_table_id": q["source_table_id"],
                  "source_row_ids": q["source_row_indices"], "kind": q["query_kind"],
                  "join_column": q["join_column"], "reason": "accepted", "policy": POLICY} for q in queries]
    write_rows(destination / "table_queryability_decisions.jsonl", decisions)
    splits = json.loads((source / "splits.json").read_text())
    splits["query_splits"] = {q["table_id"]: q["split"] for q in queries}
    save_json("splits.json", splits)
    catalog = json.loads((source / "retrieval_catalog.json").read_text())
    catalog.update(query_ids=[q["table_id"] for q in queries], query_splits=splits["query_splits"],
                   query_kinds={q["table_id"]: q["query_kind"] for q in queries},
                   qrels={q["table_id"]: q["target_table_ids"] for q in queries}, cache_namespace=destination.name)
    save_json("retrieval_catalog.json", catalog)
    explicit = {q["table_id"] for q in queries if q["query_kind"] == "explicit"}
    for name, records in (("queries", [q for q in queries if q["table_id"] in explicit]),
                          ("targets", targets), ("qrels", [r for r in qrels if r["query_table_id"] in explicit])):
        write_rows(destination / f"explicit/{name}.jsonl", records)
    for split in ("train", "dev", "test"):
        for name, records in (("queries", queries), ("qrels", qrels), ("recoveries", recoveries)):
            write_rows(destination / f"splits/{split}.{name}.jsonl", [r for r in records if r["split"] == split])
    report = json.loads((source / "BUILD.json").read_text())
    report.update(destination=str(destination), validation=validation,
                  query_decisions={"accepted": len(queries)},
                  query_context={"policy": POLICY, "source": str(source), "input_hashes": before,
                                 "changed_queries": len(changes), "added_columns": dict(Counter(c["added_column"] for c in changes)),
                                 "query_rows_splits_and_labels_preserved": True,
                                 "source_dataset_unchanged": before == dataset_hashes(source)})
    assert report["query_context"]["source_dataset_unchanged"]
    save_json("BUILD.json", report)
    save_json("VALIDATION.json", validation)
    manifest["curation"].update(cache_namespace=destination.name, query_context_policy=POLICY)
    manifest["query_construction"]["visible_context"] = {"minimum_columns": 2, "policy": POLICY}
    save_json("dataset_manifest.json", manifest)
    protocol = json.loads((source / "TRAINING_PROTOCOL.json").read_text())
    protocol.update(dataset=str(destination), stage1_directory=None, query_context_policy=POLICY)
    save_json("TRAINING_PROTOCOL.json", protocol)
    return report

"""Remove noisy join tasks and balance/resplit AbeBooks by source table."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .abebooks_ablation import _entity_reference, project_table, read_rows, write_rows

REMOVED_COLUMNS = {"availability_quantity", "seller_rating", "copy_condition_grade", "binding"}


def hash_order(values: list[str], seed: int, namespace: str) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(
        f"{seed}|{namespace}|{value}".encode()).hexdigest())


def query_kinds(qrels: list[dict]) -> dict[str, str]:
    reasons = defaultdict(set)
    for row in qrels:
        if row["rel"] > 0:
            reasons[row["query_table_id"]].add(row["reason"])
    mapping = {"model_recoverable_join_column": "implicit", "explicit_visible_join_column": "explicit"}
    if any(len(values) != 1 for values in reasons.values()):
        raise ValueError("Balance requires queries with a single implicit/explicit kind")
    return {qid: mapping[next(iter(values))] for qid, values in reasons.items()}


def grouped_split(queries: list[dict], kinds: dict[str, str], seed: int, *,
                  proportional: bool = False) -> dict[str, str]:
    """Exact class quotas with whole-source groups using a small subset-sum DP."""
    groups = defaultdict(list)
    for query in queries:
        groups[query["source_table_id"]].append(query["table_id"])
    totals = Counter(kinds.values())
    if totals["implicit"] != totals["explicit"]:
        raise ValueError("Expected balanced input")
    quota = max(1, round(totals["implicit"] * 0.1))
    # With an odd holdout size, alternate the extra class between dev/test.
    # This preserves global 50/50 while keeping the 80/10/10 query proportions.
    holdout = max(1, round(len(queries) * 0.1))
    quotas = {"dev": (holdout // 2, holdout - holdout // 2),
              "test": (holdout - holdout // 2, holdout // 2)} if proportional else {
                  "dev": (quota, quota), "test": (quota, quota)}
    remaining = set(groups)
    assignment = {}
    for split in ("dev", "test"):
        implicit_quota, explicit_quota = quotas[split]
        reachable = {(0, 0): ()}
        for source in hash_order(list(remaining), seed, split):
            counts = Counter(kinds[q] for q in groups[source])
            for (ni, ne), chosen in list(reachable.items()):
                key = (ni + counts["implicit"], ne + counts["explicit"])
                if key[0] <= implicit_quota and key[1] <= explicit_quota and key not in reachable:
                    reachable[key] = (*chosen, source)
            if (implicit_quota, explicit_quota) in reachable:
                break
        if (implicit_quota, explicit_quota) not in reachable:
            raise ValueError(f"Cannot meet {split} quotas without splitting a source group")
        chosen = reachable[implicit_quota, explicit_quota]
        for source in chosen:
            assignment[source] = split
        remaining.difference_update(chosen)
    assignment.update({source: "train" for source in remaining})
    return {q["table_id"]: assignment[q["source_table_id"]] for q in queries}


def build_balanced(source: Path, destination: Path, seed: int = 13) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    data = {name: [row for shard in spec["shards"] for row in read_rows(source / shard["path"])]
            for name, spec in manifest["artifacts"].items()}
    qrels = read_rows(source / manifest["single_files"]["qrels"])
    original_counts = {"queries": len(data["query_tables"]), "targets": len(data["data_lake_tables"]),
                       "qrels": len(qrels)}
    removed_q = {r["query_table_id"] for r in qrels
                 if r["rel"] > 0 and r["join_attribute"]["column_name"] in REMOVED_COLUMNS}
    removed_t = {t["table_id"] for t in data["data_lake_tables"]
                 if t.get("join_col_name") in REMOVED_COLUMNS}
    filtered = [r for r in qrels if r["query_table_id"] not in removed_q
                and r["target_table_id"] not in removed_t]
    kinds = query_kinds(filtered)
    before_balance = dict(Counter(kinds.values()))
    ni = before_balance["implicit"]
    explicit = hash_order([q for q, kind in kinds.items() if kind == "explicit"], seed, "balance")
    if len(explicit) < ni:
        raise ValueError("Not enough explicit queries to match all implicit queries")
    balance_removed = set(explicit[ni:])
    kept_q = set(kinds) - balance_removed
    filtered = [r for r in filtered if r["query_table_id"] in kept_q]
    kinds = query_kinds(filtered)
    data["query_tables"] = [q for q in data["query_tables"] if q["table_id"] in kept_q]
    data["data_lake_tables"] = [t for t in data["data_lake_tables"] if t["table_id"] not in removed_t]
    pairs = {(r["query_table_id"], r["target_table_id"]) for r in filtered if r["rel"] > 0}
    data["evidence_recoveries"] = [r for r in data["evidence_recoveries"]
                                  if (r["query_table_id"], r["target_table_id"]) in pairs]
    splits = grouped_split(data["query_tables"], kinds, seed)
    kept_names = {c["column_name"] for t in data["source_tables"] for c in t["columns"]} - REMOVED_COLUMNS
    maps = {t["source_table_id"]: {c["column_index"]: i for i, c in enumerate(
            c for c in t["columns"] if c["column_name"] in kept_names)} for t in data["source_tables"]}
    original_tables = {name: copy.deepcopy(data[name]) for name in
                       ("source_tables", "query_tables", "data_lake_tables")}
    for name in original_tables:
        data[name] = [project_table(t, kept_names, maps[t["source_table_id"]]) for t in data[name]]
    for query in data["query_tables"]:
        qid = query["table_id"]
        query["split"] = splits[qid]
        query["target_table_ids"] = sorted(t for q, t in pairs if q == qid)
    for row in filtered:
        row["split"] = splits[row["query_table_id"]]
        attr = row["join_attribute"]
        attr["source_column_index"] = maps[row["source_table_id"]][attr["source_column_index"]]
    for row in data["evidence_recoveries"]:
        row["split"] = splits[row["query_table_id"]]
        mapping = maps[row["source_table_id"]]
        for key in ("column_index", "source_column_index"):
            if key in row["recovered_attribute"]:
                row["recovered_attribute"][key] = mapping[row["recovered_attribute"][key]]
        _entity_reference(row["query_entity"], mapping, kept_names)
    for entity in data["entities"]:
        entity["appears_in"] = [{**r, "column_index": maps[r["source_table_id"]][r["column_index"]]}
            for r in entity["appears_in"] if r["column_index"] in maps[r["source_table_id"]]]
    for link in data["table_asset_links"]:
        link["column_index"] = maps[link["source_table_id"]].get(link["column_index"])
    for row in data["attribute_extractions"]:
        _entity_reference(row, maps[row["source_table_id"]], kept_names)
        row["candidate_attribute_names"] = [n for n in row["candidate_attribute_names"] if n in kept_names]
    # Check projection values and source/local alignment before publishing the dataset.
    source_columns = {t["source_table_id"]: t["columns"] for t in data["source_tables"]}
    for name, originals in original_tables.items():
        for old, new in zip(originals, data[name]):
            for old_row, new_row in zip(old["rows"], new["rows"]):
                old_values = {c["column_name"]: (c.get("text"), c.get("raw")) for c in old_row["cells"]}
                for cell in new_row["cells"]:
                    assert (cell.get("text"), cell.get("raw")) == old_values[cell["column_name"]]
            for i, col in enumerate(new["columns"]):
                assert col["column_index"] == i and col["column_name"] not in REMOVED_COLUMNS
                if "source_column_index" in col:
                    assert source_columns[new["source_table_id"]][col["source_column_index"]]["column_name"] == col["column_name"]
    qmap = {q["table_id"]: q for q in data["query_tables"]}
    tmap = {t["table_id"]: t for t in data["data_lake_tables"]}
    for row in filtered:
        assert row["query_table_id"] in qmap and row["target_table_id"] in tmap
        assert row["join_attribute"]["column_name"] not in REMOVED_COLUMNS
    source_splits = defaultdict(set)
    for query in data["query_tables"]:
        source_splits[query["source_table_id"]].add(query["split"])
    assert all(len(v) == 1 for v in source_splits.values())
    destination.mkdir(parents=True)
    for name, rows in data.items():
        relative = f"{name}/part-00000.jsonl"
        write_rows(destination / relative, rows)
        manifest["artifacts"][name].update(total_records=len(rows), shards=[{"path": relative, "records": len(rows)}])
    write_rows(destination / "qrels.jsonl", filtered)
    historical = {}
    for name, relative in manifest["single_files"].items():
        if name not in {"qrels", "splits"}:
            target = f"provenance/{Path(relative).name}"
            (destination / target).parent.mkdir(exist_ok=True)
            (destination / target).write_bytes((source / relative).read_bytes())
            historical[name] = target
    split_report = {split: dict(Counter(kinds[q] for q, s in splits.items() if s == split))
                    for split in ("train", "dev", "test")}
    report = {"source": str(source.resolve()), "seed": seed, "removed_columns": sorted(REMOVED_COLUMNS),
        "before": original_counts,
        "after_join_task_removal": before_balance, "after_balance": dict(Counter(kinds.values())),
        "removed_join_queries": sorted(removed_q), "removed_join_targets": sorted(removed_t),
        "downsampled_explicit_queries": sorted(balance_removed), "split_counts": split_report,
        "queries": len(qmap), "targets": len(tmap), "qrels": len(filtered),
        "evidence_recoveries": len(data["evidence_recoveries"]), "assets": len(data["bridge_assets"]),
        "kept_columns": sorted(kept_names), "source_column_maps": maps,
        "integrity": {"retained_values_unchanged": True, "source_groups_disjoint": True,
                      "dangling_qrels": 0, "removed_columns_in_tables": 0},
        "balance_policy": "query-level explicit downsampling; other surviving lake tables remain distractors",
        "historical_diagnostics": "provenance files and extraction model responses are original diagnostics, not refreshed labels"}
    manifest["single_files"] = {"qrels": "qrels.jsonl", "splits": "splits.json", **historical}
    manifest["rebalanced"] = report
    for filename, value in (("dataset_manifest.json", manifest), ("CURATION.json", report),
                            ("splits.json", {"split_key": "source_table_id", "split_policy": "query_only",
                             "data_lake_scope": "shared", "seed": seed, "counts": split_report,
                             "query_splits": splits})):
        (destination / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return report

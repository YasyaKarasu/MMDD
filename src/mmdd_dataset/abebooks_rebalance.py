"""Remove noisy join tasks and balance/resplit AbeBooks by source table."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .abebooks_ablation import _entity_reference, project_table, read_rows, write_rows
from .abebooks_curation import dataset_hashes, file_hash, load_artifacts
from .abebooks_standalone import validate_queries

REMOVED_COLUMNS = {"availability_quantity", "seller_rating", "copy_condition_grade", "binding"}


def select_balanced_standalone(queries: list[dict], seed: int = 13) -> list[dict]:
    """Keep all implicit queries; sample explicit sources in rounds within each split."""
    selected = {q["table_id"] for q in queries if q["query_kind"] == "implicit"}
    for split in ("train", "dev", "test"):
        quota = sum(q["query_kind"] == "implicit" and q["split"] == split for q in queries)
        buckets = defaultdict(list)
        for q in queries:
            if q["split"] == split and q["query_kind"] == "explicit":
                buckets[q["source_table_id"]].append(q["table_id"])
        if not quota or sum(map(len, buckets.values())) < quota:
            raise ValueError(f"Cannot retain all implicit queries and balance {split}")
        groups = hash_order(list(buckets), seed, f"{split}:sources")
        ordered = {g: hash_order(buckets[g], seed, f"{split}:{g}") for g in groups}
        candidates = [ordered[g][i] for i in range(max(map(len, ordered.values())))
                      for g in groups if i < len(ordered[g])]
        selected.update(candidates[:quota])
    return [q for q in queries if q["table_id"] in selected]


def balance_standalone(source: Path, destination: Path, seed: int = 13) -> dict:
    """Copy a qualified standalone dataset, preserving its lake, annotations and split boundaries."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be outside the source dataset")
    if destination.exists():
        raise FileExistsError(destination)
    before = dataset_hashes(source)
    manifest, data = load_artifacts(source)
    queries = select_balanced_standalone(data["query_tables"], seed)
    kept = {q["table_id"] for q in queries}
    removed = [q for q in data["query_tables"] if q["table_id"] not in kept]
    qrels = [r for r in read_rows(source / manifest["single_files"]["qrels"]) if r["query_table_id"] in kept]
    recoveries = [r for r in data["evidence_recoveries"] if r["query_table_id"] in kept]
    judgments = [r for r in read_rows(source / "audit/candidate_judgments.jsonl") if r["query_table_id"] in kept]
    source_rows = {(s["source_table_id"], r["row_id"]): r for s in data["source_tables"] for r in s["rows"]}
    validation = validate_queries(queries, data["data_lake_tables"], judgments, recoveries, source_rows)
    counts = {s: dict(Counter(q["query_kind"] for q in queries if q["split"] == s)) for s in ("train", "dev", "test")}
    assert all(c["implicit"] == c["explicit"] for c in counts.values())
    assert recoveries == data["evidence_recoveries"]
    groups = {q["split_group"]: q["split"] for q in queries}
    old_splits = json.loads((source / "splits.json").read_text())
    splits = {**old_splits, "counts": counts, "query_splits": {q["table_id"]: q["split"] for q in queries},
              "independent_components": len(groups), "component_splits": groups,
              "groups_per_split": dict(Counter(groups.values())),
              "source_groups_per_split": {s: len({q["source_table_id"] for q in queries if q["split"] == s}) for s in counts},
              "largest_component_queries": max(Counter(q["split_group"] for q in queries).values()),
              "balance_seed": seed, "balance_policy": "per_split_explicit_downsampling_round_robin_by_source",
              "original_query_assignments_preserved": True}
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".env.openai"))
    archive = destination / "provenance/before_balance"
    changed = {"query_tables", "qrels.jsonl", "splits", "splits.json", "retrieval_catalog.json",
               "explicit", "dataset_manifest.json", "BUILD.json", "REPORT.md", "VALIDATION.json",
               "TRAINING_PROTOCOL.json", "table_queryability_decisions.jsonl",
               "audit/candidate_judgments.jsonl", "audit/DELIVERY_VALIDATION.json", "audit/visual_spotchecks.jsonl"}
    for name in sorted(changed):
        old = destination / name
        if old.exists():
            archived = archive / name
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), archived)
    def save_json(name: str, value: dict) -> None:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    query_path = "query_tables/part-00000.jsonl"
    write_rows(destination / query_path, queries)
    manifest["artifacts"]["query_tables"].update(total_records=len(queries), shards=[{"path": query_path, "records": len(queries)}])
    write_rows(destination / "qrels.jsonl", qrels)
    write_rows(destination / "audit/candidate_judgments.jsonl", judgments)
    write_rows(destination / "audit/downsampled_explicit_queries.jsonl", removed)
    decisions = read_rows(source / "table_queryability_decisions.jsonl")
    removed_ids = {q["table_id"] for q in removed}
    for decision in decisions:
        if decision.get("query_table_id") in removed_ids:
            decision.update(reason="explicit_downsampled_for_per_split_balance", previous_reason=decision["reason"])
    write_rows(destination / "table_queryability_decisions.jsonl", decisions)
    reviews_path = source / "audit/visual_spotchecks.jsonl"
    if reviews_path.exists():
        reviews = read_rows(reviews_path)
        for r in reviews:
            r["active_in_balanced_dataset"] = r.get("current_query_id") in kept
        write_rows(destination / "audit/visual_spotchecks.jsonl", reviews)
    catalog = json.loads((source / "retrieval_catalog.json").read_text())
    catalog.update(query_ids=[q["table_id"] for q in queries],
                   query_kinds={q["table_id"]: q["query_kind"] for q in queries},
                   query_splits=splits["query_splits"], qrels={q["table_id"]: q["target_table_ids"] for q in queries},
                   cache_namespace=destination.name)
    save_json("retrieval_catalog.json", catalog)
    save_json("splits.json", splits)
    explicit = {q["table_id"] for q in queries if q["query_kind"] == "explicit"}
    write_rows(destination / "explicit/queries.jsonl", [q for q in queries if q["table_id"] in explicit])
    write_rows(destination / "explicit/qrels.jsonl", [r for r in qrels if r["query_table_id"] in explicit])
    write_rows(destination / "explicit/targets.jsonl", data["data_lake_tables"])
    for split in counts:
        for name, records in (("queries", queries), ("qrels", qrels), ("recoveries", recoveries)):
            write_rows(destination / f"splits/{split}.{name}.jsonl", [r for r in records if r["split"] == split])
    prior = json.loads((source / "BUILD.json").read_text())
    report = {**prior, "destination": str(destination), "queries": len(queries), "qrels": len(qrels),
              "split_counts": counts, "validation": validation,
              **{k: splits[k] for k in ("groups_per_split", "source_groups_per_split", "independent_components", "largest_component_queries")},
              "query_decisions": dict(Counter(r["reason"] for r in decisions)),
              "balance": {"source": str(source), "seed": seed, "input_hashes": before,
                          "original_queries": len(data["query_tables"]), "removed_explicit_queries": len(removed),
                          "removed_by_split": dict(Counter(q["split"] for q in removed)),
                          "policy": splits["balance_policy"], "all_implicit_preserved": True,
                          "query_inputs_ids_and_assignments_unchanged": True},
              "balanced_source_unchanged": dataset_hashes(source) == before,
              "preserved_artifacts_unchanged": all(file_hash(destination / sh["path"]) == before[sh["path"]]
                  for name, spec in manifest["artifacts"].items() if name != "query_tables" for sh in spec["shards"])}
    assert report["balanced_source_unchanged"] and report["preserved_artifacts_unchanged"]
    save_json("BUILD.json", report)
    save_json("VALIDATION.json", {**validation, "balanced_source_unchanged": True, "exact_balance_each_split": True})
    manifest["curation"].update(cache_namespace=destination.name, balance_report="BUILD.json")
    manifest["single_files"]["stats"] = "BUILD.json"
    save_json("dataset_manifest.json", manifest)
    lines = ["# AbeBooks 独立训练集：各 split 50% / 50%", "",
             f"来源：`{source}`；新副本：`{destination}`。", "",
             "| split | implicit | explicit | 合计 |", "|---|---:|---:|---:|"]
    lines.extend(f"| {s} | {c['implicit']} | {c['explicit']} | {sum(c.values())} |" for s, c in counts.items())
    lines += ["", f"保留全部 implicit，在各 split 内按源表轮流抽样 explicit，seed={seed}。共移出 {len(removed)} 个 explicit，未重复任何 query 或源行。",
              "候选湖、源记录、素材和恢复标注保持原样；保留 query 的 ID、输入、正例与 split 均不变。",
              "平衡比例按 query 计数；证据路径训练使用有恢复监督的 implicit 子集。",
              "旧文件位于 `provenance/before_balance/`，移出清单位于 `audit/downsampled_explicit_queries.jsonl`。",
              "标注仍为模型辅助标注，历史测试暴露限制继续适用。本次未执行模型训练。", ""]
    (destination / "REPORT.md").write_text("\n".join(lines))
    return report


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

"""Offline, fixed-lake AbeBooks author-join curation.

The task is completion of a captured book record, using full author-field
equality. Source identities judge executed pairs; they never execute the join.
No model client, training, fuzzy name matching, or source-table rewriting.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .abebooks_ablation import read_rows, write_rows


POLICY = "abebooks_full_author_record_completion_v1"


def author_key(value: str) -> str:
    """Conservative matching only; do not invert two comma-separated full names."""
    value = unicodedata.normalize("NFKC", str(value)).strip()
    if value.casefold() in {"", "-", "--", "0", "none", "null", "n/a", "unknown"}:
        return ""
    names = []
    for part in value.split(";"):
        # Restrict inversion to a single surname token and an unqualified name.
        match = re.fullmatch(r"([^\W\d_]+(?:[-'’][^\W\d_]+)*),\s*([\w .'-]+)", part.strip())
        if match and not re.search(r"\b(and|ed|eds|editor|jr|sr)\b", part, re.I):
            part = f"{match[2]} {match[1]}"
        names.append(part)
    value = "; ".join(names).casefold().replace("’", "'").replace("'", "")
    return " ".join(re.findall(r"[^\W_]+", value))


def cell_values(row: dict) -> dict[str, str]:
    return {c["column_name"]: str(c.get("text") or "") for c in row["cells"]}


def execute_author_join(predictions: dict[int, str], target: dict) -> set[tuple[int, int]]:
    """Execute supplied values against all target rows without consulting gold IDs."""
    return {(rid, row["row_id"]) for rid, value in predictions.items() if author_key(value)
            for row in target["rows"]
            if author_key(value) == author_key(cell_values(row).get("authors", ""))}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_hashes(root: Path) -> dict[str, str]:
    # The protected filename is excluded before any file inspection or reads.
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
            if ".env.openai" not in p.parts and p.is_file()}


def load_artifacts(root: Path) -> tuple[dict, dict[str, list[dict]]]:
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    return manifest, {name: [r for shard in spec["shards"]
                            for r in read_rows(root / shard["path"])]
                      for name, spec in manifest["artifacts"].items()}


def audit_recovery(record: dict, review: dict | None) -> dict:
    """Require a content-bound review and full-field agreement, not substring support."""
    attr = record["recovered_attribute"]
    reason = "non_author_attribute_deferred"
    if attr["column_name"] == "authors":
        reason = "unreviewed" if review is None else review["status"]
        if review and reason == "full_value_supported":
            if not author_key(attr["value"]) or author_key(attr["value"]) != author_key(review["observed_value"]):
                reason = "review_value_does_not_match_full_source_value"
    return {"recovery_id": record["recovery_id"], "query_table_id": record["query_table_id"],
            "source_table_id": record["source_table_id"], "source_row_id": record["source_row_id"],
            "asset_id": record["evidence"]["asset_id"], "attribute": attr["column_name"],
            "original_value": attr["value"], "model_value": attr.get("model_value"),
            "matching_key": author_key(attr["value"]) if attr["column_name"] == "authors" else None,
            "status": reason, "usable": reason == "full_value_supported",
            "content_review": review}


def judge_target(query: dict, target: dict, source_rows: dict) -> dict:
    """Enumerate every equality pair before judging record identity and added values."""
    qid, tid = query["table_id"], target["table_id"]
    added = sorted({c["column_name"] for c in target["columns"]}
                   - {c["column_name"] for c in query["columns"]} - {"authors"})
    tvalues = [cell_values(r) for r in target["rows"]]
    keys = [author_key(r.get("authors", "")) for r in tvalues]
    pairs = []
    for qr in query["rows"]:
        qloc = (query["source_table_id"], qr["source_row_id"])
        qsource = cell_values(source_rows[qloc])
        key = author_key(qsource.get("authors", ""))
        if not key:
            continue
        for tr, values, tkey in zip(target["rows"], tvalues, keys):
            if key != tkey:
                continue
            tloc = (target["source_table_id"], tr["source_row_id"])
            tsource = cell_values(source_rows[tloc])
            if qloc == tloc:
                status = "correct_record"
            elif not qsource.get("title") or not tsource.get("title") or author_key(qsource["title"]) == author_key(tsource["title"]):
                status = "unjudged_record_identity"
            else:
                status = "different_book_record"
            pairs.append({"query_row_id": qr["row_id"], "target_row_id": tr["row_id"],
                          "query_source_row": list(qloc), "target_source_row": list(tloc),
                          "join_key": key, "status": status,
                          "added_values": {name: values[name] for name in added if values.get(name)}})
    unknown = any(p["status"] == "unjudged_record_identity" for p in pairs)
    wrong = any(p["status"] == "different_book_record" for p in pairs)
    useful = {p["query_row_id"] for p in pairs if p["status"] == "correct_record" and p["added_values"]}
    if not pairs:
        status, reason = "negative", "no_full_author_value_overlap"
    elif wrong:
        status, reason = "negative", "author_join_expands_to_other_book_records"
    elif unknown:
        status, reason = "unjudged", "record_identity_requires_review"
    elif len(set(keys) - {""}) < 2:
        status, reason = "negative", "constant_target_author_column"
    elif not useful:
        status, reason = "negative", "no_nonempty_missing_attribute_added"
    elif len(useful) < 2:
        status, reason = "unjudged", "only_one_useful_row_requires_review"
    else:
        status, reason = "positive", "all_executed_pairs_correct_and_at_least_two_useful_rows"
    return {"query_table_id": qid, "target_table_id": tid, "status": status,
            "reason": reason, "added_columns": added, "useful_query_rows": sorted(useful),
            "matched_query_rows": len({p["query_row_id"] for p in pairs}),
            "matching_pairs": len(pairs), "row_pairs": pairs}


def visible_author_leaks(query: dict, source_rows: dict) -> list[dict]:
    """Check actual cell input for full values or explicitly delimited author names."""
    visible = [(r["row_id"], c["column_name"], author_key(c.get("text", "")))
               for r in query["rows"] for c in r["cells"]]
    leaks = []
    for row in query["rows"]:
        value = cell_values(source_rows[query["source_table_id"], row["source_row_id"]]).get("authors", "")
        names = {author_key(value), *(author_key(p) for p in value.split(";"))} - {""}
        for rid, column, text in visible:
            for name in sorted(names):
                if len(name) >= 4 and f" {name} " in f" {text} ":
                    leaks.append({"hidden_row_id": row["row_id"], "visible_row_id": rid,
                                  "column_name": column, "matching_key": name})
    return leaks


def split_audit(queries: list[dict], source_rows: dict, assets: list[dict],
                recoveries: list[dict]) -> dict:
    """Audit source groups, record/title identities, and exact/near evidence copies."""
    groups = defaultdict(lambda: defaultdict(set))
    by_location = defaultdict(set)
    qmap = {q["table_id"]: q for q in queries}
    for q in queries:
        groups["source_group"][q["source_table_id"]].add(q["split"])
        for row in q["rows"]:
            loc = (q["source_table_id"], row["source_row_id"])
            values = cell_values(source_rows[loc])
            by_location[loc].add(q["split"])
            groups["source_row"][str(loc)].add(q["split"])
            groups["title_author"][str((author_key(values.get("title", "")), author_key(values.get("authors", ""))))].add(q["split"])
            for c in source_rows[loc]["cells"]:
                if c.get("wiki_title"):
                    groups["entity_id"][c["wiki_title"]].add(q["split"])
    approved_splits = defaultdict(set)
    for r in recoveries:
        if r["query_table_id"] in qmap:
            approved_splits[r["evidence"]["asset_id"]].add(qmap[r["query_table_id"]]["split"])
    image_bits, text_shingles = {}, {}
    for a in assets:
        attached = by_location.get((a.get("source_table_id"), a.get("source_row_id")), set())
        approved = approved_splits[a["asset_id"]]
        if not attached and not approved:
            continue
        if a["asset_type"] == "image":
            from PIL import Image
            with Image.open(a["local_path"]) as im:
                rgb = im.convert("RGB")
                content = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                thumb = rgb.convert("L").resize((9, 8))
                pixels = list(thumb.getdata())
                bits = sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
                           for y in range(8) for x in range(8))
                image_bits[a["asset_id"]] = (bits, attached)
        else:
            tokens = re.findall(r"\w+", a["content"].casefold())
            content = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
            if len(tokens) >= 20:
                text_shingles[a["asset_id"]] = ({tuple(tokens[i:i + 5]) for i in range(len(tokens) - 4)}, attached)
        groups["attached_evidence_content"][content].update(attached)
        groups["approved_evidence_content"][content].update(approved)
    near = []
    for modality, items in (("image", image_bits), ("text", text_shingles)):
        ids = sorted(items)
        for i, aid in enumerate(ids):
            va, sa = items[aid]
            for bid in ids[i + 1:]:
                vb, sb = items[bid]
                if not sa or not sb or len(sa | sb) < 2:
                    continue
                similarity = 1 - (va ^ vb).bit_count() / 64 if modality == "image" else len(va & vb) / len(va | vb)
                if similarity >= (0.96875 if modality == "image" else 0.9):
                    near.append({"asset_ids": [aid, bid], "modality": modality,
                                 "splits": sorted(sa | sb), "similarity": similarity,
                                 "status": "possible_duplicate_not_confirmed"})
    return {"cross_split": {kind: [{"key": key, "splits": sorted(splits)} for key, splits in values.items()
                                    if len(splits) > 1] for kind, values in groups.items()},
            "near_duplicate_candidates": near,
            "independent_source_groups": len(groups["source_group"]),
            "scope": "Query rows and their attached evidence; the complete retrieval corpus is intentionally shared.",
            "near_duplicate_policy": "image dHash <=2 bits; text word-5-gram Jaccard >=0.90; flags are diagnostic",
            "historically_exposed_test": True}


def curate_dataset(source: Path, destination: Path, reviews_path: Path) -> dict:
    """Create a physical metadata copy and curate only that new directory."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be outside the source dataset")
    if destination.exists():
        raise FileExistsError(destination)
    before = dataset_hashes(source)
    manifest, data = load_artifacts(source)
    original_qrels = read_rows(source / manifest["single_files"]["qrels"])
    reviews = {r["recovery_id"]: r for r in read_rows(reviews_path)}
    assets = {a["asset_id"]: a for a in data["bridge_assets"]}
    recoveries = {r["recovery_id"]: r for r in data["evidence_recoveries"]}
    for rid, review in reviews.items():
        record = recoveries[rid]
        asset = assets[record["evidence"]["asset_id"]]
        content_hash = (file_hash(Path(asset["local_path"])) if asset["asset_type"] == "image"
                        else hashlib.sha256(asset["content"].encode()).hexdigest())
        if review["asset_id"] != asset["asset_id"] or review["content_sha256"] != content_hash:
            raise ValueError(f"Review content changed: {rid}")
    evidence_audit = [audit_recovery(r, reviews.get(r["recovery_id"])) for r in recoveries.values()]
    usable = {r["recovery_id"] for r in evidence_audit if r["usable"]}
    source_rows = {(s["source_table_id"], r["row_id"]): r for s in data["source_tables"] for r in s["rows"]}
    # Every target cell must really be the claimed source projection.
    for target in data["data_lake_tables"]:
        if not target.get("rows") or not target.get("columns"):
            raise ValueError(f"Empty target: {target['table_id']}")
        for row in target["rows"]:
            original = cell_values(source_rows[target["source_table_id"], row["source_row_id"]])
            if any(original.get(k) != v for k, v in cell_values(row).items()):
                raise ValueError(f"Target provenance mismatch: {target['table_id']}")
    rels_by_query = defaultdict(list)
    for rel in original_qrels:
        rels_by_query[rel["query_table_id"]].append(rel)
    queries, qrels, active_recoveries = [], [], []
    query_audit, judgments, oracle = [], [], []
    for original in data["query_tables"]:
        q = copy.deepcopy(original)
        qid = q["table_id"]
        old_rels = rels_by_query[qid]
        author_rels = [r for r in old_rels if r["join_attribute"]["column_name"] == "authors"]
        kind = "implicit" if any(r["reason"] == "model_recoverable_join_column" for r in old_rels) else "explicit"
        audit = {"query_table_id": qid, "source_table_id": q["source_table_id"], "split": q["split"],
                 "kind": kind, "original_attributes": sorted({r["join_attribute"]["column_name"] for r in old_rels}),
                 "status": "excluded", "reasons": []}
        if not author_rels:
            audit["reasons"].append("non_author_task_deferred")
            query_audit.append(audit)
            continue
        values = [cell_values(source_rows[q["source_table_id"], r["source_row_id"]]).get("authors", "") for r in q["rows"]]
        keys = [author_key(v) for v in values]
        audit["source_values"] = values
        audit["matching_keys"] = keys
        if not all(keys):
            audit["reasons"].append("invalid_or_placeholder_author_in_query")
        if len(set(keys) - {""}) < 2:
            audit["reasons"].append("constant_query_author_column")
        leaks = visible_author_leaks(q, source_rows) if kind == "implicit" else []
        audit["visible_leaks"] = leaks
        if leaks:
            audit["reasons"].append("visible_author_leak")
        approved = [r for r in recoveries.values() if r["query_table_id"] == qid and r["recovery_id"] in usable]
        supported_rows = {r["query_row_id"] for r in approved}
        audit["supported_row_ids"] = sorted(supported_rows)
        audit["recovery_coverage"] = len(supported_rows) / len(q["rows"]) if kind == "implicit" else None
        if kind == "implicit" and len(supported_rows) < 2:
            audit["reasons"].append("fewer_than_two_full_value_supported_rows")
        scanned = [judge_target(q, t, source_rows) for t in data["data_lake_tables"]]
        judgments.extend(scanned)
        audit["judgment_counts"] = dict(Counter(j["status"] for j in scanned))
        positive = [j for j in scanned if j["status"] == "positive"]
        if not positive:
            audit["reasons"].append("no_useful_unambiguous_author_join")
        if any(j["status"] == "unjudged" for j in scanned):
            audit["reasons"].append("incomplete_candidate_judgments")
        if kind == "implicit" and any(len(supported_rows & set(j["useful_query_rows"])) < 2 for j in positive):
            audit["reasons"].append("positive_has_fewer_than_two_supported_rows")
        audit["confirmed_positive_ids"] = [j["target_table_id"] for j in positive]
        if audit["reasons"]:
            query_audit.append(audit)
            continue
        audit["status"] = "retained"
        audit["recovery_status"] = ("partial_recovery_verifiable" if len(supported_rows) < len(q["rows"]) else "all_rows_verifiable") if kind == "implicit" else "visible_author_join"
        q["target_table_ids"] = audit["confirmed_positive_ids"]
        q["curation"] = {"policy": POLICY, "original_query_id": qid, "recovery_status": audit["recovery_status"],
                         "supported_row_ids": sorted(supported_rows), "judged_coverage": 1.0}
        q["hidden_attributes"] = [h for h in q.get("hidden_attributes", []) if h["column_name"] == "authors"]
        for hidden in q["hidden_attributes"]:
            hidden.update(recovered_rows=len(supported_rows), recovered_value_ratio=len(supported_rows) / len(q["rows"]))
        queries.append(q)
        query_audit.append(audit)
        for j in positive:
            rel = copy.deepcopy(author_rels[0])
            rel.update(target_table_id=j["target_table_id"], data_lake_table_id=j["target_table_id"],
                       curation_policy=POLICY, judgment="positive", row_pair_count=j["matching_pairs"],
                       added_columns=j["added_columns"])
            rel["join_attribute"].update(recovered_rows=len(supported_rows), selected_rows=len(q["rows"]),
                                         recovered_value_ratio=len(supported_rows) / len(q["rows"]))
            qrels.append(rel)
            counts = Counter(p["query_row_id"] for p in j["row_pairs"])
            replay = {r["query_row_id"]: reviews[r["recovery_id"]]["observed_value"] for r in approved}
            target = next(t for t in data["data_lake_tables"] if t["table_id"] == j["target_table_id"])
            replay_pairs = execute_author_join(replay, target)
            # Fixed cyclic swap on the judged rows; record accidental answer matches.
            row_ids = sorted(replay)
            swapped = dict(zip(row_ids, [replay[r] for r in row_ids[1:] + row_ids[:1]]))
            swap_pairs = execute_author_join(swapped, target)
            gold_pairs = {(p["query_row_id"], p["target_row_id"]) for p in j["row_pairs"]}
            oracle.append({"query_table_id": qid, "target_table_id": j["target_table_id"], "kind": kind,
                           "oracle_pairs": j["matching_pairs"], "oracle_correct_pairs": j["matching_pairs"],
                           "oracle_query_row_coverage": len(counts) / len(q["rows"]),
                           "matches_per_row": dict(counts), "approved_value_replay_pairs": len(replay_pairs),
                           "approved_value_replay_correct_pairs": len(replay_pairs & gold_pairs),
                           "approved_value_replay_query_row_coverage": len({rid for rid, _ in replay_pairs}) / len(q["rows"]),
                           "replayed_values": replay,
                           "judged_recovery_rows": len(replay), "no_evidence_author_join_pairs": 0 if kind == "implicit" else j["matching_pairs"],
                           "swapped_evidence_pairs": len(swap_pairs),
                           "swapped_evidence_correct_pairs": len(swap_pairs & gold_pairs),
                           "scope": "Offline content-review value replay, not fresh model recovery or retrieval results."})
            for record in approved:
                pair_ids = [p["target_row_id"] for p in j["row_pairs"] if p["query_row_id"] == record["query_row_id"]]
                if not pair_ids:
                    continue
                r = copy.deepcopy(record)
                r["curation"] = {"policy": POLICY, "original_recovery_id": record["recovery_id"],
                                 "observed_value": reviews[record["recovery_id"]]["observed_value"]}
                r.update(target_table_id=j["target_table_id"], data_lake_table_id=j["target_table_id"], target_row_ids=pair_ids)
                r["path_nodes"][-1]["node_id"] = j["target_table_id"]
                if j["target_table_id"] != record["target_table_id"]:
                    suffix = hashlib.sha256(j["target_table_id"].encode()).hexdigest()[:12]
                    r["recovery_id"] += f"_curated_{suffix}"
                    r["path_id"] += f"_curated_{suffix}"
                active_recoveries.append(r)
    split_report = split_audit(queries, source_rows, data["bridge_assets"], active_recoveries)
    for kind in ("source_group", "source_row", "entity_id", "title_author", "approved_evidence_content"):
        if split_report["cross_split"].get(kind):
            raise ValueError(f"Retained queries have cross-split {kind} duplicates; review before export")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".env.openai"))
    history = destination / "provenance" / "pre_author_curation"
    changed_files = ["dataset_manifest.json", "qrels.jsonl", "splits.json", "retrieval_catalog.json",
                     "table_queryability_decisions.jsonl", "REPORT.md", "VALIDATION.json", "REGENERATION.json"]
    for name in ("query_tables", "evidence_recoveries"):
        changed_files.extend(s["path"] for s in manifest["artifacts"][name]["shards"])
    changed_files.extend(str(p.relative_to(destination)) for p in (destination / "explicit").glob("*.jsonl"))
    for name in changed_files:
        old = destination / name
        if old.exists():
            archived = history / name
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), archived)
    def save_json(name: str, value: dict) -> None:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    for name, records in (("query_tables", queries), ("evidence_recoveries", active_recoveries)):
        path = f"{name}/part-00000.jsonl"
        write_rows(destination / path, records)
        manifest["artifacts"][name] = {"directory": name, "total_records": len(records),
                                      "max_records_per_shard": 50000, "shards": [{"path": path, "records": len(records)}]}
    write_rows(destination / "qrels.jsonl", qrels)
    write_rows(destination / "table_queryability_decisions.jsonl", query_audit)
    write_rows(destination / "audit/query_audit.jsonl", query_audit)
    write_rows(destination / "audit/candidate_judgments.jsonl", judgments)
    write_rows(destination / "audit/oracle_and_replay.jsonl", oracle)
    write_rows(destination / "audit/negative_sampling_exclusions.jsonl", [
        {"query_table_id": a["query_table_id"], "status": a["status"],
         "forbidden_negative_ids": [j["target_table_id"] for j in judgments
                                    if j["query_table_id"] == a["query_table_id"] and j["status"] != "negative"]}
        for a in query_audit if "judgment_counts" in a])
    write_rows(destination / "audit/evidence_content_reviews.jsonl", list(reviews.values()))
    retained_ids = {q["table_id"] for q in queries}
    for audit in evidence_audit:
        audit["used_in_main_dataset"] = audit["usable"] and audit["query_table_id"] in retained_ids
        audit["unused_reason"] = (None if audit["used_in_main_dataset"] else
                                  "query_excluded" if audit["usable"] else audit["status"])
    write_rows(destination / "audit/evidence_audit.jsonl", evidence_audit)
    save_json("audit/split_audit.json", split_report)
    variants = defaultdict(set)
    normalization = []
    for (sid, rid), row in source_rows.items():
        value = cell_values(row).get("authors")
        if value is not None:
            key = author_key(value)
            variants[key].add(value)
            normalization.append({"source_table_id": sid, "source_row_id": rid, "original": value,
                                  "matching_key": key, "valid": bool(key)})
    write_rows(destination / "audit/author_matching_keys.jsonl", normalization)
    save_json("audit/normalization_collisions.json", {k: sorted(v) for k, v in variants.items() if k and len(v) > 1})
    kinds = {a["query_table_id"]: a["kind"] for a in query_audit}
    splits = {q["table_id"]: q["split"] for q in queries}
    split_counts = {s: dict(Counter(kinds[qid] for qid, split in splits.items() if split == s)) for s in ("train", "dev", "test")}
    old_splits = json.loads((source / "splits.json").read_text())
    save_json("splits.json", {**old_splits, "counts": split_counts, "query_splits": splits,
                              "curation_policy": "original_assignments_preserved_no_rebalancing"})
    catalog = json.loads((source / "retrieval_catalog.json").read_text())
    catalog["query_ids"] = list(splits)
    catalog["query_kinds"] = {qid: kinds[qid] for qid in splits}
    catalog["query_splits"] = splits
    catalog["qrels"] = {q["table_id"]: q["target_table_ids"] for q in queries}
    catalog["numerical_index_status"] = "requires_fresh_embedding_and_ANN_build"
    save_json("retrieval_catalog.json", catalog)
    explicit_ids = {qid for qid in splits if kinds[qid] == "explicit"}
    write_rows(destination / "explicit/queries.jsonl", [q for q in queries if q["table_id"] in explicit_ids])
    write_rows(destination / "explicit/targets.jsonl", data["data_lake_tables"])
    write_rows(destination / "explicit/qrels.jsonl", [r for r in qrels if r["query_table_id"] in explicit_ids])
    summary = {"policy": POLICY, "source": str(source), "destination": str(destination),
               "task": "Complete captured book records using conservative full-author-field equality; no other-book expansion.",
               "source_tables": len(data["source_tables"]), "source_rows": len(source_rows),
               "candidate_tables": len(data["data_lake_tables"]), "assets": len(assets),
               "original_queries": len(data["query_tables"]), "retained_queries": len(queries),
               "author_implicit_queries_audited": sum(a["kind"] == "implicit" and "authors" in a["original_attributes"] for a in query_audit),
               "split_counts": split_counts, "qrels": len(qrels),
               "added_positive_pairs": len({(r["query_table_id"], r["target_table_id"]) for r in qrels}
                                           - {(r["query_table_id"], r["target_table_id"]) for r in original_qrels}),
               "original_recoveries": len(recoveries), "active_recovery_paths": len(active_recoveries),
               "distinct_active_facts": len({(r["source_table_id"], r["source_row_id"], r["recovered_attribute"]["value"]) for r in active_recoveries}),
               "evidence_audit_counts": dict(Counter(a["status"] for a in evidence_audit)),
               "query_exclusion_counts": dict(Counter(r for a in query_audit for r in a["reasons"])),
               "all_retained_candidates_judged": True,
               "judged_query_target_pairs": len(queries) * len(data["data_lake_tables"]),
               "retained_fully_recoverable_implicit_queries": sum(a.get("recovery_status") == "all_rows_verifiable" for a in query_audit),
               "candidate_hashes_unchanged": all(file_hash(destination / s["path"]) == before[s["path"]]
                                                for s in manifest["artifacts"]["data_lake_tables"]["shards"]),
               "original_dataset_unchanged": before == dataset_hashes(source),
               "preserved_artifact_hashes_unchanged": all(
                   file_hash(destination / shard["path"]) == before[shard["path"]]
                   for name, spec in manifest["artifacts"].items()
                   if name not in {"query_tables", "evidence_recoveries"} for shard in spec["shards"]),
               "input_hashes": before, "reviews_sha256": file_hash(reviews_path),
               "image_policy": "Original external image files remain shared read-only; no image content or asset paths edited.",
               "limitations": ["Historical splits are exposed, not independent unseen test data.",
                               "Unannotated rows have oracle source values, not verified recoverability.",
                               "Value replay is a diagnostic, not fresh model extraction or a Direct retrieval baseline.",
                               "Negative judgments are scoped to the full-author equality record-completion task."]}
    if not all(summary[k] for k in ("original_dataset_unchanged", "candidate_hashes_unchanged", "preserved_artifact_hashes_unchanged")):
        raise ValueError("Input or candidate preservation check failed")
    save_json("CURATION.json", summary)
    save_json("VALIDATION.json", {k: summary[k] for k in ("original_dataset_unchanged", "candidate_hashes_unchanged", "all_retained_candidates_judged")})
    manifest["single_files"]["stats"] = "CURATION.json"
    manifest["query_construction"] = {"provider": "abebooks", "policy_version": POLICY,
                                       "history": "provenance/pre_author_curation/dataset_manifest.json"}
    manifest.pop("explicit_regeneration", None)
    manifest["curation"] = {"report": "CURATION.json", "query_audit": "audit/query_audit.jsonl",
                            "candidate_judgments": "audit/candidate_judgments.jsonl",
                            "all_active_candidates_judged": True, "cache_namespace": destination.name}
    save_json("dataset_manifest.json", manifest)
    retained_audits = [a for a in query_audit if a["status"] == "retained"]
    implicit_oracle = [o for o in oracle if o["kind"] == "implicit"]
    lines = ["# AbeBooks 作者连接审核副本", "", f"来源：`{source}`。策略：`{POLICY}`。", "",
             "任务是对当前抓取的书目记录补充字段：按完整作者字段做保守规范化等值连接，执行结果必须全部对应正确记录，且至少两行得到非空增补字段。来源 ID 仅用于审核行对，不是连接键或模型输入。",
             "", f"保留 {summary['candidate_tables']} 张候选表、{summary['source_tables']} 张源表、{summary['source_rows']} 条源记录和 {summary['assets']} 个素材；这些文件与输入逐字节相同。",
             f"主任务保留 {len(queries)} 个查询、{len(qrels)} 条正例、{summary['distinct_active_facts']} 个不同恢复事实。所有原查询与标注保存在 `provenance/pre_author_curation/`。",
             "", "| 划分 | implicit | explicit |", "|---|---:|---:|"]
    lines.extend(f"| {s} | {counts.get('implicit', 0)} | {counts.get('explicit', 0)} |" for s, counts in split_counts.items())
    lines += ["", "沿用原划分，没有补齐类型比例。测试集已在历史实验中暴露，只适合描述性机制验证；未审核的查询行不计作属性恢复成功或失败。",
              "", f"对保留查询完成 {summary['judged_query_target_pairs']} 个 query-target 判断。新增正例 {summary['added_positive_pairs']} 个；只在严格任务定义下将其他候选判断为负例。未判定候选及正例写入 `audit/negative_sampling_exclusions.jsonl`，有未判定目标的查询不进入主任务。",
              "", "## 离线连接验证", "",
              f"隐式查询原值补回：{sum(o['oracle_correct_pairs'] for o in implicit_oracle)}/{sum(o['oracle_pairs'] for o in implicit_oracle)} 个正确行对。",
              f"内容复核值回放：{sum(o['approved_value_replay_correct_pairs'] for o in implicit_oracle)}/{sum(o['approved_value_replay_pairs'] for o in implicit_oracle)} 个正确行对；仅覆盖已复核行。",
              f"固定循环交换复核值：{sum(o['swapped_evidence_correct_pairs'] for o in implicit_oracle)}/{sum(o['swapped_evidence_pairs'] for o in implicit_oracle)} 个正确行对。",
              "这里的回放与无证据/交换对照只验证等值连接逻辑，不代表新模型恢复、Direct 检索或端到端性能。没有新增模型调用或训练。",
              "", "## 保留查询", "", "| Query | 类型 | split | 已复核行 |", "|---|---|---|---|"]
    lines.extend(f"| {a['query_table_id']} | {a['kind']} | {a['split']} | {a['supported_row_ids']} |" for a in retained_audits)
    cross = split_report["cross_split"]
    lines += ["", "## 文件与限制", "",
              "- `audit/query_audit.jsonl`：逐查询决定、排除原因、原值和规范化键。",
              "- `audit/evidence_audit.jsonl`：168 条历史恢复标注的使用状态；58 条作者证据进行了内容复核，审核者是 Codex，并非独立人工标注。",
              "- `audit/candidate_judgments.jsonl`：作者查询全湖判断、连接行对、增补值及依据。",
              "- `audit/oracle_and_replay.jsonl`：逐例 Oracle、复核值回放与交换对照。",
              "- `audit/split_audit.json`：实体、行、证据内容及近重复检查。",
              f"- 查询源组、源行、实体、书名作者组合及批准证据未发现跨划分重复；附属素材仍有 {len(cross.get('attached_evidence_content', []))} 个跨划分重复内容组、{len(split_report['near_duplicate_candidates'])} 对近重复候选，保留并披露，未按模型表现删素材。",
              "- 没有把原作者字段重写成外部真实作者全集；未判定的作者别名、缺失作者、编辑角色不能宣称已修复。",
              "- 外部图片路径保持原样，共享图片只读。不要原地编辑共享图片；若需要修改图像内容，另行复制图片。",
              "- 新训练应重新构建 Stage-1 数据与缓存，旧 qrels、训练列表和 checkpoint 不应作为本版本的新训练结果。",
              "- 多正例评测区分 Recall@k 与 Hit@k；本次严格审核未发现需要新增的合格正例。", ""]
    (destination / "REPORT.md").write_text("\n".join(lines))
    return summary

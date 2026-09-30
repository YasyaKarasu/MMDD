"""Regenerate explicit joins on sources that never produced implicit queries."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from .abebooks_ablation import read_rows, write_rows
from .abebooks_rebalance import hash_order, query_kinds


def assert_disjoint_sources(qrels: list[dict]) -> dict[str, set[str]]:
    kinds = query_kinds(qrels)
    sources = {kind: {r["source_table_id"] for r in qrels
                      if r["rel"] > 0 and kinds[r["query_table_id"]] == kind}
               for kind in ("implicit", "explicit")}
    overlap = sources["implicit"] & sources["explicit"]
    if overlap:
        raise ValueError(f"implicit/explicit source overlap: {sorted(overlap)}")
    return sources


def historical_implicit_sources(decisions: list[dict], qrels: list[dict]) -> set[str]:
    """Keep historical successes excluded even if their queries were later pruned."""
    sources = {r["source_table_id"] for r in qrels
               if r["rel"] > 0 and r["reason"] == "model_recoverable_join_column"}
    for decision in decisions:
        original = decision.get("rejected_multimodal_decision") or {}
        if (decision.get("reason") == "queryable"
                or decision.get("rejected_multimodal_reason") == "queryable"
                or original.get("reason") == "queryable"):
            sources.add(decision["source_table_id"])
    return sources


def select_candidates(candidates: dict[str, list[dict]], quotas: dict[str, int],
                      seed: int) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Assign whole sources, then select variants round-robin within each split.

    Split source quotas follow query proportions; extra sources go to a split
    only when necessary for capacity. A shortfall never borrows implicit sources.
    """
    remaining = hash_order(list(candidates), seed, "explicit-source-splits")
    total = sum(quotas.values())
    n_sources = len(remaining)
    source_splits = {}
    selected = {}
    for split in ("dev", "test", "train"):
        needed = quotas.get(split, 0)
        count = (len(remaining) if split == "train" else
                 min(len(remaining), max(int(needed > 0), round(n_sources * needed / max(1, total)))))
        assigned, remaining = remaining[:count], remaining[count:]
        while sum(len(candidates[s]) for s in assigned) < needed and remaining:
            assigned.append(remaining.pop(0))
        queues = {s: hash_order([c["candidate_id"] for c in candidates[s]], seed, split)
                  for s in assigned}
        by_id = {c["candidate_id"]: c for s in assigned for c in candidates[s]}
        picked = 0
        while picked < needed and any(queues.values()):
            for source in assigned:
                if queues[source] and picked < needed:
                    selected.setdefault(source, []).append(by_id[queues[source].pop(0)])
                    source_splits[source] = split
                    picked += 1
    return selected, source_splits


def regenerate_explicit(source: Path, destination: Path, seed: int = 13) -> dict:
    """Preserve implicit records, replace explicit records, and rebuild catalogs."""
    if destination.exists():
        raise FileExistsError(destination)
    # Reuse the maintained construction core, without constructing model clients.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts_old"))
    import build_mm_joinability_dataset as builder
    from clean_abebooks_joinability import modal_share

    manifest = json.loads((source / "dataset_manifest.json").read_text())
    original_manifest = copy.deepcopy(manifest)
    data = {name: [r for shard in spec["shards"] for r in read_rows(source / shard["path"])]
            for name, spec in manifest["artifacts"].items()}
    original_qrels = read_rows(source / manifest["single_files"]["qrels"])
    history_path = manifest.get("explicit_regeneration", {}).get(
        "historical_decisions", manifest["single_files"]["table_queryability_decisions"])
    decisions = read_rows(source / history_path)
    kinds = query_kinds(original_qrels)
    implicit_ids = {qid for qid, kind in kinds.items() if kind == "implicit"}
    implicit_queries = [q for q in data["query_tables"] if q["table_id"] in implicit_ids]
    implicit_qrels = [r for r in original_qrels if r["query_table_id"] in implicit_ids]
    implicit_recoveries = data["evidence_recoveries"]
    if any(r["query_table_id"] not in implicit_ids for r in implicit_recoveries):
        raise ValueError("Unexpected evidence recoveries for a non-implicit query")
    excluded = historical_implicit_sources(decisions, original_qrels)
    source_tables = {t["source_table_id"]: t for t in data["source_tables"]}
    decision_by_source = {d["source_table_id"]: d for d in decisions}
    quotas = dict(Counter(q["split"] for q in implicit_queries))
    args = SimpleNamespace(seed=seed, query_rows_per_table=5,
                           min_rows_per_output_table=2, min_column_non_empty_ratio=0.5,
                           max_query_tables_per_source_table=0)
    candidates = {}
    rejected = []
    degenerate = []
    for sid, table in source_tables.items():
        if sid in excluded:
            continue
        if sid not in decision_by_source:
            raise ValueError(f"Missing historical implicit decision for {sid}")
        entity_col = (table["metadata"].get("candidate_entity_columns") or [None])[0]
        generated = builder.build_explicit_join_fallback_candidates(
            source_table=table, split="train", entity_col=entity_col,
            rejected_multimodal_reason=decision_by_source[sid]["reason"],
            args=args, force=True)
        viable = []
        for candidate in generated:
            if modal_share(table, candidate["join_column_index"]) >= 1.0:
                degenerate.append({"source_table_id": sid,
                                   "column_name": candidate["join_column_name"]})
            else:
                viable.append(candidate)
        if viable:
            candidates[sid] = viable
        else:
            rejected.append({"source_table_id": sid, "reason": (
                "no_retained_entity_column" if entity_col is None else "no_viable_nonconstant_join")})
    selected, source_splits = select_candidates(candidates, quotas, seed)
    explicit_queries, explicit_targets, explicit_qrels, new_decisions = [], [], [], []
    for sid, group in selected.items():
        split = source_splits[sid]
        rebuilt = builder.rebuild_selected_explicit_join_candidates(
            source_table=source_tables[sid], split=split, candidate_decisions=group, args=args)
        for candidate in rebuilt:
            queries, targets, qrels, _ = builder.materialize_balanced_explicit_join_candidate(
                source_table=source_tables[sid], split=split, candidate_decision=candidate, args=args)
            if len(queries) != 1:
                raise ValueError(f"Selected explicit candidate did not materialize: {sid}")
            explicit_queries.extend(queries)
            explicit_targets.extend(targets)
            explicit_qrels.extend(qrels)
        new_decisions.append({"source_table_id": sid, "reason": "explicit_join_fallback",
                              "explicit_join_candidates": rebuilt})
    # A versioned namespace prevents stale ID-keyed embeddings from being mistaken
    # for new projections after the source column indices/context have changed.
    namespace = hashlib.sha256(json.dumps({
        "queries": explicit_queries, "replaced_query_ids": sorted(set(kinds) - implicit_ids),
    }, sort_keys=True).encode()).hexdigest()[:12]
    id_map = {r["table_id"]: f'{r["table_id"]}_disjoint_{namespace}'
              for r in [*explicit_queries, *explicit_targets]}

    def rekey(value):
        if isinstance(value, str):
            return id_map.get(value, value)
        if isinstance(value, list):
            return [rekey(item) for item in value]
        if isinstance(value, dict):
            return {key: rekey(item) for key, item in value.items()}
        return value

    explicit_queries, explicit_targets, explicit_qrels, new_decisions = rekey(
        [explicit_queries, explicit_targets, explicit_qrels, new_decisions])
    old_targets = data["data_lake_tables"]
    old_explicit_ids = {t["table_id"] for t in old_targets
                        if t.get("construction_type") == "explicit_visible_join"
                        or str(t.get("chain_id", "")).startswith("chain_explicit_")}
    old_explicit_ids.update(r["target_table_id"] for r in original_qrels
                            if kinds[r["query_table_id"]] == "explicit")
    # Preserve existing implicit target projections, including historical unlabelled
    # ones, but rebuild raw distractors only for sources with no surviving target.
    retained_targets = [t for t in old_targets if t["table_id"] not in old_explicit_ids
                        and t["role"] == "target_data_lake_table"]
    targets = [*retained_targets, *explicit_targets]
    represented = {t["source_table_id"] for t in targets}
    restored = []
    for sid, table in source_tables.items():
        if sid in represented:
            continue
        columns = [c["column_index"] for c in table["columns"]]
        rows, row_ids = builder.project_selected_rows(
            table, columns, {r["row_id"] for r in table["rows"]}, min_required_cols=0)
        restored.append(builder.table_record(
            table_id=f"dl_raw_{sid}", role="raw_data_lake_table", split=None,
            source_table=table, column_indices=columns, rows=rows,
            source_row_indices=row_ids, extra={"queryable_source_table": False}))
    targets.extend(restored)
    data["query_tables"] = [*implicit_queries, *explicit_queries]
    data["data_lake_tables"] = targets
    qrels = [*implicit_qrels, *explicit_qrels]
    sources = assert_disjoint_sources(qrels)
    if sources["explicit"] & excluded:
        raise ValueError("Explicit source succeeded at implicit construction historically")
    qmap = {q["table_id"]: q for q in data["query_tables"]}
    tmap = {t["table_id"]: t for t in targets}
    if len(qmap) != len(data["query_tables"]) or len(tmap) != len(targets):
        raise ValueError("Duplicate table IDs")
    if set(tmap) & old_explicit_ids:
        raise ValueError("Obsolete explicit targets survived")
    assert {t["source_table_id"] for t in targets} == set(source_tables)
    gold = defaultdict(set)
    for rel in qrels:
        query, target = qmap[rel["query_table_id"]], tmap[rel["target_table_id"]]
        assert query["source_table_id"] == target["source_table_id"] == rel["source_table_id"]
        assert query["split"] == rel["split"]
        gold[query["table_id"]].add(target["table_id"])
    for query in qmap.values():
        assert set(query["target_table_ids"]) == gold[query["table_id"]]
    for recovery in implicit_recoveries:
        assert recovery["target_table_id"] in gold[recovery["query_table_id"]]
    query_splits = {qid: q["split"] for qid, q in qmap.items()}
    source_split_sets = defaultdict(set)
    for query in qmap.values():
        source_split_sets[query["source_table_id"]].add(query["split"])
    assert all(len(splits) == 1 for splits in source_split_sets.values())
    new_kinds = query_kinds(qrels)
    split_counts = {split: dict(Counter(new_kinds[qid] for qid, s in query_splits.items() if s == split))
                    for split in ("train", "dev", "test")}
    report = {
        "source": str(source.resolve()), "seed": seed,
        "policy": "exclude all historical implicit successes, preserve implicit queries and splits",
        "excluded_historical_implicit_sources": sorted(excluded),
        "eligible_sources": len(source_tables) - len(excluded),
        "candidate_sources": len(candidates),
        "candidate_queries": sum(map(len, candidates.values())),
        "noncandidate_sources": rejected, "constant_join_candidates_removed": degenerate,
        "implicit_queries": len(implicit_queries), "explicit_queries": len(explicit_queries),
        "implicit_sources": len(sources["implicit"]), "explicit_sources": len(sources["explicit"]),
        "shared_sources": 0, "split_counts": split_counts,
        "shortfall_by_split": {s: quotas.get(s, 0) - split_counts[s].get("explicit", 0)
                               for s in ("train", "dev", "test")},
        "removed_explicit_queries": sorted(set(kinds) - implicit_ids),
        "removed_explicit_targets": sorted(old_explicit_ids),
        "targets": len(targets), "new_explicit_targets": len(explicit_targets),
        "retained_implicit_targets": len(retained_targets), "raw_lake_tables": len(restored),
        "qrels": len(qrels), "evidence_recoveries": len(implicit_recoveries),
        "assets": len(data["bridge_assets"]),
        "integrity": {"implicit_queries_qrels_recoveries_unchanged": True,
                      "source_splits_disjoint": True, "obsolete_explicit_targets_remaining": 0,
                      "dangling_qrels": 0, "lake_source_coverage": len(source_tables)},
        "downstream": "ID catalog rebuilt; embeddings, ANN indexes and models must be rebuilt for this dataset",
    }
    destination.mkdir(parents=True)
    for name, rows in data.items():
        relative = f"{name}/part-00000.jsonl"
        write_rows(destination / relative, rows)
        manifest["artifacts"][name].update(total_records=len(rows), shards=[{"path": relative, "records": len(rows)}])
    write_rows(destination / "qrels.jsonl", qrels)
    write_rows(destination / "provenance/original_decisions.jsonl", decisions)
    # Current decisions never present historical column indices as current ones.
    current_decisions = [{"source_table_id": sid, "reason": "retained_implicit",
                          "query_table_ids": [q["table_id"] for q in implicit_queries if q["source_table_id"] == sid]}
                         for sid in sorted(sources["implicit"])] + new_decisions
    write_rows(destination / "table_queryability_decisions.jsonl", current_decisions)
    for name, records in (("queries", explicit_queries), ("targets", explicit_targets), ("qrels", explicit_qrels)):
        write_rows(destination / f"explicit/{name}.jsonl", records)
    manifest["single_files"] = {"qrels": "qrels.jsonl", "splits": "splits.json",
                                "stats": "REGENERATION.json", "table_queryability_decisions": "table_queryability_decisions.jsonl"}
    manifest.pop("rebalanced", None)
    manifest["explicit_regeneration"] = {"report": "REGENERATION.json", "historical_decisions": "provenance/original_decisions.jsonl"}
    manifest["query_construction"]["explicit_source_policy"] = "never_successful_implicit"
    values = {
        "dataset_manifest.json": manifest, "REGENERATION.json": report,
        "provenance/original_manifest.json": original_manifest,
        "splits.json": {"split_key": "source_table_id", "split_policy": "query_only",
                        "data_lake_scope": "shared", "seed": seed, "counts": split_counts,
                        "query_splits": query_splits},
        "retrieval_catalog.json": {"query_ids": list(qmap), "target_ids": list(tmap),
                                   "query_kinds": new_kinds, "query_splits": query_splits,
                                   "qrels": {qid: sorted(ids) for qid, ids in gold.items()},
                                   "numerical_index_status": "requires_fresh_embedding_and_ANN_build"},
    }
    for name, value in values.items():
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return report

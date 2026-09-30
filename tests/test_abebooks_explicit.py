import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmdd_dataset.abebooks_ablation import read_rows, write_rows
from mmdd_dataset.abebooks_explicit import (
    assert_disjoint_sources, historical_implicit_sources, regenerate_explicit, select_candidates,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_old"))
import build_abebooks_mm_joinability_dataset as abebooks
import build_mm_joinability_dataset as builder


def source_table(sid):
    names = ["title", "authors", "publisher", "price"]
    return {"source_table_id": sid, "provenance_builder": "abebooks_mm_joinability_dataset",
            "source_file": "synthetic", "num_rows": 7, "num_cols": len(names),
            "columns": [{"column_index": i, "column_name": n} for i, n in enumerate(names)],
            "metadata": {"candidate_entity_columns": [0]},
            "rows": [{"row_id": r, "cells": [
                {"column_index": i, "column_name": n, "text": f"{n}-{r}", "raw": f"{n}-{r}"}
                for i, n in enumerate(names)]} for r in range(7)]}


def relation(qid, tid, sid, kind="implicit"):
    return {"query_table_id": qid, "target_table_id": tid, "source_table_id": sid,
            "rel": 3, "split": "train", "reason": (
                "model_recoverable_join_column" if kind == "implicit" else "explicit_visible_join_column"),
            "join_attribute": {"source_column_index": 1, "column_name": "authors"}}


def test_source_guard_and_historical_successes_survive_pruning():
    rows = [relation("i", "ti", "used"), relation("e", "te", "used", "explicit")]
    with pytest.raises(ValueError, match="source overlap"):
        assert_disjoint_sources(rows)
    decisions = [{"source_table_id": "pruned", "reason": "queryable"},
                 {"source_table_id": "rejected", "reason": "no_column_met_recovered_value_ratio"}]
    assert historical_implicit_sources(decisions, rows) == {"pruned", "used"}


def test_candidate_shortfall_keeps_sources_whole_and_does_not_duplicate_queries():
    candidates = {f"s{i}": [{"candidate_id": f"c{i}_{j}"} for j in range(2)] for i in range(3)}
    selected, splits = select_candidates(candidates, {"train": 10, "dev": 1, "test": 1}, 13)
    ids = [c["candidate_id"] for group in selected.values() for c in group]
    assert len(ids) == len(set(ids)) == 4
    assert Counter(splits.values()) == {"dev": 1, "test": 1, "train": 1}
    assert (selected, splits) == select_candidates(dict(reversed(list(candidates.items()))),
                                                  {"train": 10, "dev": 1, "test": 1}, 13)


def test_live_builder_uses_only_rejected_sources_and_partitions_siblings_together(tmp_path, monkeypatch):
    tables = [source_table("implicit"), source_table("rejected")]
    args, lake_args = abebooks.parse_args([
        "--lake_dir", str(tmp_path / "lake"), "--output_dir", str(tmp_path / "out"),
        "--cache_dir", str(tmp_path / "cache"), "--explicit_join_fallback_mode", "match_implicit"])
    monkeypatch.setattr(abebooks, "prepare_abebooks", lambda *a, **k: SimpleNamespace(
        source_tables=tables, entities=[], skipped={}))
    monkeypatch.setattr(abebooks, "adapt_assets", lambda *a, **k: [])
    monkeypatch.setattr(builder, "source_splits", lambda *a: ({}, {t["source_table_id"]: "train" for t in tables}))
    monkeypatch.setattr(builder, "auto_check_required", lambda _: False)
    real_generate = builder.build_explicit_join_fallback_candidates

    def only_rejected(**kwargs):
        assert kwargs["source_table"]["source_table_id"] == "rejected"
        return real_generate(**kwargs)

    monkeypatch.setattr(builder, "build_explicit_join_fallback_candidates", only_rejected)

    def construct(**kwargs):
        table = kwargs["source_table"]
        if table["source_table_id"] == "implicit":
            return ([{"table_id": f"i{j}"} for j in range(2)], [],
                    [relation(f"i{j}", f"ti{j}", "implicit") for j in range(2)], {"reason": "queryable"})
        candidates = only_rejected(source_table=table, split="train", entity_col=0,
                                   rejected_multimodal_reason="no_recovery", args=args, force=True)
        return [], [builder.raw_data_lake_record(table)], [], {
            "reason": "no_recovery", "explicit_join_candidates": candidates}

    monkeypatch.setattr(builder, "build_table_join_records", construct)
    abebooks.build_dataset(args, lake_args, extractor=object())
    qrels = read_rows(tmp_path / "out/qrels.jsonl")
    assert assert_disjoint_sources(qrels) == {"implicit": {"implicit"}, "explicit": {"rejected"}}
    queries = read_rows(tmp_path / "out/query_tables/part-00000.jsonl")
    explicit = [q for q in queries if q.get("construction_type") == "explicit_visible_join"]
    join_names = {q["join_col_name"] for q in explicit}
    assert len(explicit) == 2
    assert all(not join_names.intersection(q["query_context_col_names"]) for q in explicit)


def test_regeneration_replaces_orphan_targets_and_preserves_implicit(tmp_path):
    source = tmp_path / "old"
    destination = tmp_path / "new"
    implicit = {"table_id": "qi", "source_table_id": "used", "split": "train", "target_table_ids": ["ti"]}
    old_explicit = {"table_id": "qe", "source_table_id": "used", "split": "train", "target_table_ids": ["te"]}
    target = {"table_id": "ti", "source_table_id": "used", "role": "target_data_lake_table"}
    data = {
        "source_tables": [source_table(s) for s in ("used", "pruned", "rejected")],
        "query_tables": [implicit, old_explicit],
        "data_lake_tables": [target, {"table_id": "te", "source_table_id": "used",
            "role": "target_data_lake_table", "construction_type": "explicit_visible_join"},
            {"table_id": "orphan", "source_table_id": "pruned", "role": "target_data_lake_table",
             "construction_type": "explicit_visible_join"}],
        "evidence_recoveries": [{"query_table_id": "qi", "target_table_id": "ti"}],
        "bridge_assets": [], "entities": [], "attribute_extractions": [], "table_asset_links": [],
    }
    manifest = {"artifacts": {}, "single_files": {"qrels": "qrels.jsonl",
                "table_queryability_decisions": "decisions.jsonl"}, "query_construction": {}}
    for name, rows in data.items():
        path = f"{name}/part-00000.jsonl"
        write_rows(source / path, rows)
        manifest["artifacts"][name] = {"shards": [{"path": path, "records": len(rows)}], "total_records": len(rows)}
    original_qrels = [relation("qi", "ti", "used"), relation("qe", "te", "used", "explicit")]
    write_rows(source / "qrels.jsonl", original_qrels)
    write_rows(source / "decisions.jsonl", [
        {"source_table_id": s, "reason": "queryable" if s != "rejected" else "no_column_met_recovered_value_ratio"}
        for s in ("used", "pruned", "rejected")])
    (source / "dataset_manifest.json").write_text(json.dumps(manifest))
    before = copy.deepcopy(data)
    report = regenerate_explicit(source, destination)
    assert report["implicit_queries"] == report["explicit_queries"] == 1
    assert report["removed_explicit_targets"] == ["orphan", "te"]
    assert report["excluded_historical_implicit_sources"] == ["pruned", "used"]
    assert report["integrity"]["lake_source_coverage"] == 3
    queries = read_rows(destination / "query_tables/part-00000.jsonl")
    assert queries[0] == before["query_tables"][0]
    assert queries[1]["source_table_id"] == "rejected"
    assert "_disjoint_" in queries[1]["table_id"]
    lake = read_rows(destination / "data_lake_tables/part-00000.jsonl")
    assert lake[0] == target
    assert {t["source_table_id"] for t in lake} == {"used", "pruned", "rejected"}
    assert not {"te", "orphan"} & {t["table_id"] for t in lake}
    assert read_rows(destination / "qrels.jsonl")[0] == original_qrels[0]
    assert read_rows(destination / "evidence_recoveries/part-00000.jsonl") == before["evidence_recoveries"]
    # Reruns must still see pruned historical successes, not only current queries.
    second = regenerate_explicit(destination, tmp_path / "second")
    assert second["excluded_historical_implicit_sources"] == ["pruned", "used"]

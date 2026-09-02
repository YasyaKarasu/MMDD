import argparse
import json
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

import build_mm_table_dataset as mm_table_dataset
import build_mm_joinability_dataset as join_dataset
import build_stage1_embeddings as stage1_embeddings
import qwen3_vl_embedding
from build_stage1_evidence_paths import run as build_evidence_paths
from build_stage1_logic_connectivity import add_same_source_bridge_positive_pairs, build_query_table_splits, build_target_rows, choose_query_context_cols
from build_stage1_logic_connectivity import run as build_logic_connectivity
from build_mm_table_dataset import ShardedJsonlWriter, WikipediaClient, build_bridge_assets, split_text_asset_content
from hitl_annotation_app import create_app
from generate_weak_labels import label_path
from stage1_connection_viewer import create_app as create_connection_viewer_app
from stage1_connection_viewer import load_assets
from stage1_connection_viewer import load_groups
from stage1_recall_viewer import assemble_query_cards, load_recorded_recall_cards, render_targets
from merge_human_labels import run as merge_human_labels
from select_hitl_batch import run as select_hitl_batch
from qwen3_vl_embedding import Qwen3VLEmbeddingEncoder, image_limit_from_exception, resize_batch_images_for_limit
from stage1_gui import format_gui_urls, resolve_gui_host
from stage1_io import (
    fd_purity,
    iter_jsonl,
    iter_manifest_records,
    project_rows,
    write_jsonl,
)
from stage1_serialization import (
    serialize_image_asset_prompt,
    serialize_table_for_embedding,
    serialize_text_asset_for_embedding,
)
from stage1_training_cache import clear_training_outputs
from train_teacher import TeacherMLP, score_paths
from train_student import Student, build_distill_records, build_ranking_groups, train_loss
from wikimedia_media import MediaFailureRecorder, MediaPolicyConfig, NonImageMediaError
from eval_stage1_recall import (
    beam_search_tables,
    bridge_recall_record,
    direct_eval_qrels,
    direct_recall_records,
    infer_raw_embedding_hnsw,
    infer_table_only,
    recall_fraction,
    relation_query_from_projected,
    table_rankings,
)


def cell(idx, name, text):
    return {
        "column_index": idx,
        "column_name": name,
        "text": text,
        "wiki_title": text if idx == 0 else None,
        "has_wiki_link": idx == 0,
    }


def synthetic_table():
    data = [
        ("Messi", "Argentina", "Forward", "36"),
        ("Haaland", "Norway", "Forward", "23"),
        ("Mbappe", "France", "Forward", "25"),
        ("Modric", "Croatia", "Midfielder", "38"),
        ("Kane", "England", "Forward", "30"),
        ("Son", "South Korea", "Forward", "31"),
    ]
    rows = [
        {
            "row_id": row_id,
            "cells": [
                cell(0, "Player", row[0]),
                cell(1, "Country", row[1]),
                cell(2, "Position", row[2]),
                cell(3, "Age", row[3]),
            ],
        }
        for row_id, row in enumerate(data)
    ]
    return {
        "source_table_id": "st_synth",
        "page_title": "Football players",
        "caption": "",
        "section_title": "",
        "columns": [
            {"column_index": 0, "column_name": "Player"},
            {"column_index": 1, "column_name": "Country"},
            {"column_index": 2, "column_name": "Position"},
            {"column_index": 3, "column_name": "Age"},
        ],
        "rows": rows,
        "metadata": {
            "candidate_entity_columns": [0],
            "column_profiles": [
                {"column_index": 0, "column_name": "Player", "non_empty_ratio": 1.0, "unique_ratio": 1.0, "numeric_ratio": 0.0},
                {"column_index": 1, "column_name": "Country", "non_empty_ratio": 1.0, "unique_ratio": 1.0, "numeric_ratio": 0.0},
                {"column_index": 2, "column_name": "Position", "non_empty_ratio": 1.0, "unique_ratio": 0.33, "numeric_ratio": 0.0},
                {"column_index": 3, "column_name": "Age", "non_empty_ratio": 1.0, "unique_ratio": 1.0, "numeric_ratio": 1.0},
            ],
        },
    }


def test_fd_purity():
    table = synthetic_table()
    purity, support = fd_purity(table["rows"], 0, 1)
    assert support == 6
    assert purity == pytest.approx(1.0)


def test_abc_fragment_provenance_core():
    table = synthetic_table()
    q_rows, q_source_rows = project_rows(table, [0, 1], dedupe_col=0)
    t_rows, t_source_rows = build_target_rows(table, 1, [2])
    assert len(q_rows) == 6
    assert len(t_rows) == 6
    assert q_source_rows == [0, 1, 2, 3, 4, 5]
    assert t_source_rows == [0, 1, 2, 3, 4, 5]
    assert q_rows[0]["cells"][0]["source_column_index"] == 0


def test_query_fragments_keep_irrelevant_context_without_target_leakage():
    table = synthetic_table()
    context_cols = choose_query_context_cols(table, {0, 1, 2}, max_cols=2)
    assert context_cols == [3]

    visible_cols = [0, 1] + context_cols
    hidden_cols = [0] + context_cols
    qv_rows, _ = project_rows(table, visible_cols, dedupe_col=0, min_required_cols=2)
    qh_rows, _ = project_rows(table, hidden_cols, dedupe_col=0, min_required_cols=1)

    assert [cell["source_column_index"] for cell in qv_rows[0]["cells"]] == [0, 1, 3]
    assert [cell["source_column_index"] for cell in qh_rows[0]["cells"]] == [0, 3]
    assert 2 not in [cell["source_column_index"] for cell in qv_rows[0]["cells"]]
    assert 1 not in [cell["source_column_index"] for cell in qh_rows[0]["cells"]]


def test_project_rows_sanitizes_url_cells_in_output_text():
    table = {
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "Reference"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    cell(0, "Entity", "Alpha"),
                    cell(1, "Reference", "https://example.com/" + "a" * 200),
                ],
            },
            {
                "row_id": 1,
                "cells": [
                    cell(0, "Entity", "Beta"),
                    cell(1, "Reference", "see https://example.org/ref?token=" + "b" * 160 + " mirror"),
                ],
            },
        ],
    }

    rows, _source_rows = project_rows(table, [0, 1], min_required_cols=1)

    assert rows[0]["cells"][1]["text"] == "[url]"
    assert rows[1]["cells"][1]["text"] == "see mirror"


def test_table_only_logic_connectivity_does_not_create_hidden_queries(tmp_path):
    input_dir = tmp_path / "input"
    stage = tmp_path / "stage"
    rows = []
    for row_id, (entity, bridge, attr) in enumerate(
        [
            ("a1", "b1", "c1"),
            ("a2", "b1", "c1"),
            ("a3", "b1", "c1"),
            ("a4", "b2", "c2"),
            ("a5", "b2", "c2"),
            ("a6", "b2", "c2"),
        ]
    ):
        rows.append(
            {
                "row_id": row_id,
                "cells": [
                    cell(0, "Entity", entity),
                    cell(1, "Bridge", bridge),
                    cell(2, "Attr", attr),
                ],
            }
        )
    table = {
        "source_table_id": "source_1",
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "Bridge"},
            {"column_index": 2, "column_name": "Attr"},
        ],
        "rows": rows,
        "metadata": {"candidate_entity_columns": [0]},
    }
    write_jsonl(input_dir / "source_tables" / "part-00000.jsonl", [table])
    (input_dir / "dataset_manifest.json").write_text(
        json.dumps({"artifacts": {"source_tables": {"shards": [{"path": "source_tables/part-00000.jsonl", "records": 1}]}}}),
        encoding="utf-8",
    )
    (input_dir / "splits.json").write_text(json.dumps({"train": {"source_table_ids": ["source_1"]}}), encoding="utf-8")

    build_logic_connectivity(
        argparse.Namespace(
            input_dir=str(input_dir),
            output_dir=str(stage),
            min_rows_per_fragment=2,
            max_chains_per_table=10,
            max_bridges_per_anchor=5,
            max_target_attrs=1,
            max_query_context_attrs=0,
            seed=13,
            min_ab_purity=0.95,
            min_bc_purity=0.85,
            min_support=2,
            max_bridge_unique_ratio=0.85,
            table_only=True,
        )
    )

    fragments = [json.loads(line) for line in (stage / "logic_fragments.jsonl").read_text().splitlines()]
    qrels = [json.loads(line) for line in (stage / "qrels.jsonl").read_text().splitlines()]
    pairs = [json.loads(line) for line in (stage / "logic_pairs.jsonl").read_text().splitlines()]
    assert {fragment["role"] for fragment in fragments} == {"left_visible", "right_target"}
    assert {qrel["query_role"] for qrel in qrels} == {"left_visible"}
    fragment_roles = {fragment["fragment_id"]: fragment["role"] for fragment in fragments}
    assert {fragment_roles[pair["query_fragment_id"]] for pair in pairs} == {"left_visible"}


def test_table_only_query_corpus_mode_marks_targets_as_shared_corpus(tmp_path):
    input_dir = tmp_path / "input"
    stage = tmp_path / "stage"
    rows = []
    for row_id, (entity, bridge, attr) in enumerate(
        [
            ("a1", "b1", "c1"),
            ("a2", "b1", "c1"),
            ("a3", "b1", "c1"),
            ("a4", "b2", "c2"),
            ("a5", "b2", "c2"),
            ("a6", "b2", "c2"),
        ]
    ):
        rows.append(
            {
                "row_id": row_id,
                "cells": [
                    cell(0, "Entity", entity),
                    cell(1, "Bridge", bridge),
                    cell(2, "Attr", attr),
                ],
            }
        )
    table = {
        "source_table_id": "source_1",
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "Bridge"},
            {"column_index": 2, "column_name": "Attr"},
        ],
        "rows": rows,
        "metadata": {"candidate_entity_columns": [0]},
    }
    (input_dir / "source_tables").mkdir(parents=True)
    write_jsonl(input_dir / "source_tables" / "part-00000.jsonl", [table])
    (input_dir / "dataset_manifest.json").write_text(
        json.dumps({"artifacts": {"source_tables": {"shards": [{"path": "source_tables/part-00000.jsonl", "records": 1}]}}}),
        encoding="utf-8",
    )
    (input_dir / "splits.json").write_text(json.dumps({"train": {"source_table_ids": ["source_1"]}}), encoding="utf-8")

    build_logic_connectivity(
        argparse.Namespace(
            input_dir=str(input_dir),
            output_dir=str(stage),
            min_rows_per_fragment=2,
            max_chains_per_table=10,
            max_bridges_per_anchor=5,
            max_target_attrs=1,
            max_query_context_attrs=0,
            seed=13,
            min_ab_purity=0.95,
            min_bc_purity=0.85,
            min_support=2,
            max_bridge_unique_ratio=0.85,
            table_only=True,
            data_lake_split_mode="query_corpus",
        )
    )

    fragments = [json.loads(line) for line in (stage / "logic_fragments.jsonl").read_text().splitlines()]
    qrels = [json.loads(line) for line in (stage / "qrels.jsonl").read_text().splitlines()]
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))

    assert {fragment["split"] for fragment in fragments if fragment["role"] == "left_visible"} == {"train"}
    assert {fragment["split"] for fragment in fragments if fragment["role"] == "right_target"} == {"corpus"}
    assert {qrel["split"] for qrel in qrels} == {"train"}
    assert manifest["logic_connectivity"]["data_lake_split_mode"] == "query_corpus"


def test_webtable_mode_builds_table_only_fragments_from_csv_benchmark(tmp_path):
    input_dir = tmp_path / "webtable"
    table_dir = input_dir / "data" / "benchmark" / "webtable" / "large" / "split_1"
    stage = tmp_path / "stage"
    table_dir.mkdir(parents=True)
    (input_dir / "webtable_join_query.csv").write_text(
        "query_table,query_column\nquery.csv,Player\n",
        encoding="utf-8",
    )
    (input_dir / "webtable_join_ground_truth.csv").write_text(
        "query_table,candidate_table,query_column,candidate_column\n"
        "query.csv,candidate.csv,Player,Name\n",
        encoding="utf-8",
    )
    (table_dir / "query.csv").write_text(
        "Player,Team,Age\n"
        "Messi,Inter Miami,36\n"
        "Morgan,San Diego,34\n",
        encoding="utf-8",
    )
    (table_dir / "candidate.csv").write_text(
        "Name,Country,Club\n"
        "Messi,Argentina,Inter Miami\n"
        "Morgan,USA,San Diego\n",
        encoding="utf-8",
    )

    build_logic_connectivity(
        argparse.Namespace(
            input_dir=str(input_dir),
            output_dir=str(stage),
            min_rows_per_fragment=1,
            max_chains_per_table=10,
            max_bridges_per_anchor=5,
            max_target_attrs=1,
            max_query_context_attrs=1,
            seed=13,
            min_ab_purity=0.95,
            min_bc_purity=0.85,
            min_support=1,
            max_bridge_unique_ratio=0.85,
            table_only=False,
            webtable_mode=True,
            webtable_query_file=None,
            webtable_ground_truth_file=None,
            webtable_table_dir=None,
            webtable_max_rows=20,
            webtable_recursive_lookup=False,
            webtable_split_ratios=[0.7, 0.1, 0.2],
            webtable_split_seed=13,
        )
    )

    fragments = [json.loads(line) for line in (stage / "logic_fragments.jsonl").read_text().splitlines()]
    qrels = [json.loads(line) for line in (stage / "qrels.jsonl").read_text().splitlines()]
    pairs = [json.loads(line) for line in (stage / "logic_pairs.jsonl").read_text().splitlines()]
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))

    assert {fragment["role"] for fragment in fragments} == {"left_visible", "right_target"}
    assert len([fragment for fragment in fragments if fragment["role"] == "left_visible"]) == 1
    assert len([fragment for fragment in fragments if fragment["role"] == "right_target"]) == 2
    assert len(qrels) == 1
    assert qrels[0]["query_role"] == "left_visible"
    assert qrels[0]["webtable_query_column"] == "Player"
    assert qrels[0]["webtable_candidate_columns"] == ["Name"]
    assert len(pairs) == 1
    assert manifest["logic_connectivity"]["dataset_mode"] == "webtable"
    assert manifest["logic_connectivity"]["table_only"] is True


def test_webtable_mode_splits_by_query_table(tmp_path):
    input_dir = tmp_path / "webtable"
    table_dir = input_dir / "data" / "benchmark" / "webtable" / "large" / "split_1"
    stage = tmp_path / "stage"
    table_dir.mkdir(parents=True)
    query_lines = ["query_table,query_column"]
    truth_lines = ["query_table,candidate_table,query_column,candidate_column"]
    for idx in range(5):
        query_name = f"query_{idx}.csv"
        candidate_name = f"candidate_{idx}.csv"
        query_lines.append(f"{query_name},Key")
        truth_lines.append(f"{query_name},{candidate_name},Key,Key")
        (table_dir / query_name).write_text("Key,Value\na,1\nb,2\n", encoding="utf-8")
        (table_dir / candidate_name).write_text("Key,Other\na,x\nb,y\n", encoding="utf-8")
    (input_dir / "webtable_join_query.csv").write_text("\n".join(query_lines) + "\n", encoding="utf-8")
    (input_dir / "webtable_join_ground_truth.csv").write_text("\n".join(truth_lines) + "\n", encoding="utf-8")

    build_logic_connectivity(
        argparse.Namespace(
            input_dir=str(input_dir),
            output_dir=str(stage),
            min_rows_per_fragment=1,
            max_chains_per_table=10,
            max_bridges_per_anchor=5,
            max_target_attrs=1,
            max_query_context_attrs=1,
            seed=13,
            min_ab_purity=0.95,
            min_bc_purity=0.85,
            min_support=1,
            max_bridge_unique_ratio=0.85,
            table_only=False,
            webtable_mode=True,
            webtable_query_file=None,
            webtable_ground_truth_file=None,
            webtable_table_dir=None,
            webtable_max_rows=20,
            webtable_recursive_lookup=False,
            webtable_split_ratios=[0.6, 0.2, 0.2],
            webtable_split_seed=13,
        )
    )

    fragments = [json.loads(line) for line in (stage / "logic_fragments.jsonl").read_text().splitlines()]
    qrels = [json.loads(line) for line in (stage / "qrels.jsonl").read_text().splitlines()]
    split_payload = json.loads((stage / "webtable_splits.json").read_text(encoding="utf-8"))

    query_splits = [fragment["split"] for fragment in fragments if fragment["role"] == "left_visible"]
    target_splits = {fragment["split"] for fragment in fragments if fragment["role"] == "right_target"}
    assert {split: query_splits.count(split) for split in ("train", "dev", "test")} == {"train": 3, "dev": 1, "test": 1}
    assert target_splits == {"corpus"}
    assert {split: sum(1 for qrel in qrels if qrel["split"] == split) for split in ("train", "dev", "test")} == {
        "train": 3,
        "dev": 1,
        "test": 1,
    }
    assert split_payload["counts"] == {"train": 3, "dev": 1, "test": 1}


def test_webtable_strict_mode_splits_targets_and_drops_cross_split_qrels(tmp_path):
    input_dir = tmp_path / "webtable"
    table_dir = input_dir / "data" / "benchmark" / "webtable" / "large" / "split_1"
    stage = tmp_path / "stage"
    table_dir.mkdir(parents=True)
    ratios = [0.5, 0.0, 0.5]
    query_candidate_pairs = [(f"query_{idx}.csv", f"candidate_{idx}.csv") for idx in range(20)]
    split_map, _ = build_query_table_splits({name for pair in query_candidate_pairs for name in pair}, ratios, 13)
    same_split_pairs = [(query, candidate) for query, candidate in query_candidate_pairs if split_map[query] == split_map[candidate]]
    cross_split_pairs = [(query, candidate) for query, candidate in query_candidate_pairs if split_map[query] != split_map[candidate]]
    assert same_split_pairs
    assert cross_split_pairs

    query_lines = ["query_table,query_column"]
    truth_lines = ["query_table,candidate_table,query_column,candidate_column"]
    for query_name, candidate_name in query_candidate_pairs:
        query_lines.append(f"{query_name},Key")
        truth_lines.append(f"{query_name},{candidate_name},Key,Key")
        (table_dir / query_name).write_text("Key,Value\na,1\nb,2\n", encoding="utf-8")
        (table_dir / candidate_name).write_text("Key,Other\na,x\nb,y\n", encoding="utf-8")
    (input_dir / "webtable_join_query.csv").write_text("\n".join(query_lines) + "\n", encoding="utf-8")
    (input_dir / "webtable_join_ground_truth.csv").write_text("\n".join(truth_lines) + "\n", encoding="utf-8")

    build_logic_connectivity(
        argparse.Namespace(
            input_dir=str(input_dir),
            output_dir=str(stage),
            min_rows_per_fragment=1,
            max_chains_per_table=10,
            max_bridges_per_anchor=5,
            max_target_attrs=1,
            max_query_context_attrs=1,
            seed=13,
            min_ab_purity=0.95,
            min_bc_purity=0.85,
            min_support=1,
            max_bridge_unique_ratio=0.85,
            table_only=False,
            webtable_mode=True,
            data_lake_split_mode="strict",
            webtable_query_file=None,
            webtable_ground_truth_file=None,
            webtable_table_dir=None,
            webtable_max_rows=20,
            webtable_recursive_lookup=False,
            webtable_split_ratios=ratios,
            webtable_split_seed=13,
        )
    )

    fragments = [json.loads(line) for line in (stage / "logic_fragments.jsonl").read_text().splitlines()]
    qrels = [json.loads(line) for line in (stage / "qrels.jsonl").read_text().splitlines()]
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    target_by_id = {fragment["fragment_id"]: fragment for fragment in fragments if fragment["role"] == "right_target"}

    assert len(qrels) == len(same_split_pairs)
    assert manifest["logic_connectivity"]["data_lake_split_mode"] == "strict"
    assert manifest["logic_connectivity"]["counts"]["webtable_split_mismatch_qrels"] == len(cross_split_pairs)
    assert "corpus" not in {target["split"] for target in target_by_id.values()}
    assert all(target_by_id[qrel["target_id"]]["split"] == qrel["split"] for qrel in qrels)


def test_same_source_targets_with_query_bridge_column_are_positive():
    def frag(fragment_id, role, source_cols, chain_id="c1", source_table_id="s1", **extra):
        return {
            "fragment_id": fragment_id,
            "role": role,
            "split": "train",
            "chain_id": chain_id,
            "source_table_id": source_table_id,
            "source_column_indices": source_cols,
            **extra,
        }

    fragments = [
        frag("q_visible", "left_visible", [0, 1], visible_bridge=True, visible_bridge_col=1, visible_bridge_col_name="Country"),
        frag("q_hidden", "left_hidden", [0], hidden_bridge_col=1, hidden_bridge_col_name="Country"),
        frag("target_original", "right_target", [1, 2]),
        frag("target_other_chain", "right_target", [3, 1], chain_id="c2"),
        frag("target_no_bridge", "right_target", [2, 3], chain_id="c3"),
        frag("target_other_source", "right_target", [1, 4], chain_id="c4", source_table_id="s2"),
    ]
    pairs = [
        {
            "query_fragment_id": "q_hidden",
            "target_fragment_id": "target_original",
            "label": 1,
        }
    ]
    qrels = [
        {
            "query_id": "q_hidden",
            "target_id": "target_original",
            "rel": 2,
        }
    ]

    added = add_same_source_bridge_positive_pairs(fragments, pairs, qrels)

    positive_pairs = {(pair["query_fragment_id"], pair["target_fragment_id"]) for pair in pairs}
    positive_qrels = {(qrel["query_id"], qrel["target_id"]) for qrel in qrels}
    assert added == 3
    assert ("q_visible", "target_original") in positive_pairs
    assert ("q_visible", "target_other_chain") in positive_pairs
    assert ("q_hidden", "target_other_chain") in positive_pairs
    assert ("q_hidden", "target_original") in positive_pairs
    assert ("q_hidden", "target_no_bridge") not in positive_pairs
    assert ("q_hidden", "target_other_source") not in positive_pairs
    assert ("q_hidden", "target_other_chain") in positive_qrels


def test_table_serialization_excludes_hidden_fields_and_values():
    table = synthetic_table()
    hidden_fragment = {
        **table,
        "columns": [{"column_index": 0, "source_column_index": 0, "column_name": "Player"}],
        "rows": [
            {"row_id": row["row_id"], "cells": [row["cells"][0]]}
            for row in table["rows"]
        ],
        "role": "left_hidden",
        "chain_id": "chain_secret",
        "hidden_bridge_col": 1,
        "hidden_bridge_col_name": "Country",
        "source_column_indices": [0],
        "source_row_indices": [0, 1],
        "statement": "Player -> hidden(Country)",
        "label": 1,
    }
    text = serialize_table_for_embedding(hidden_fragment)
    assert "left_hidden" not in text
    assert "chain_secret" not in text
    assert "hidden(Country)" not in text
    assert "Football players" not in text
    assert "Argentina" not in text
    assert "Country" not in text
    assert "Player" in text


def test_table_serialization_excludes_shared_page_context():
    table = {
        **synthetic_table(),
        "page_title": "Shared Wikipedia Page",
        "title": "Shared Title",
        "caption": "Shared Caption",
        "section_title": "Shared Section",
        "source_table_id": "shared_source",
    }

    text = serialize_table_for_embedding(table)

    assert "Shared Wikipedia Page" not in text
    assert "Shared Title" not in text
    assert "Shared Caption" not in text
    assert "Shared Section" not in text
    assert "shared_source" not in text
    assert "Player" in text
    assert "Messi" in text


def test_table_serialization_sanitizes_cell_urls_for_model_context():
    table = {
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "Reference"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    cell(0, "Entity", "Alpha"),
                    cell(1, "Reference", "https://example.com/" + "a" * 200),
                ],
            },
            {
                "row_id": 1,
                "cells": [
                    cell(0, "Entity", "Beta"),
                    cell(1, "Reference", "Official page: https://example.org/wiki/Beta?token=" + "b" * 160 + " archived"),
                ],
            },
        ],
    }

    text = serialize_table_for_embedding(table)

    assert "[url]" in text
    assert "Official page: archived" in text
    assert "example.com" not in text
    assert "example.org" not in text


def test_text_asset_serialization_exposes_only_intrinsic_content():
    serialized = serialize_text_asset_for_embedding(
        {
            "content": "Alpha is located in Texas.",
            "entity_wiki_title": "SECRET_WIKIPEDIA_TITLE",
            "source": "SECRET_SOURCE",
            "url": "https://private.example/wiki/Alpha",
            "asset_id": "SECRET_ASSET_ID",
        }
    )

    assert serialized == "[Text Material]\nAlpha is located in Texas."
    assert "SECRET" not in serialized
    assert "private.example" not in serialized


def test_image_asset_prompt_exposes_no_asset_metadata_or_relationship():
    prompt = serialize_image_asset_prompt(
        {
            "entity_wiki_title": "SECRET_WIKIPEDIA_TITLE",
            "source": "SECRET_SOURCE",
            "file_name": "SECRET_FILE.jpg",
            "description_url": "https://private.example/file",
            "metadata": {
                "extmetadata": {
                    "ObjectName": {"value": "SECRET_OBJECT"},
                    "ImageDescription": {"value": "SECRET_DESCRIPTION"},
                    "Categories": {"value": "SECRET_CATEGORY"},
                    "Credit": {"value": "SECRET_CREDIT"},
                }
            },
        }
    )

    assert prompt == "Represent only the intrinsic visual content of this independent image."
    assert "SECRET" not in prompt
    assert "private.example" not in prompt


def test_weak_labels_ignore_asset_provenance_and_image_metadata():
    path = {
        "entity_text": "Alpha",
        "bridge_value": "Texas",
        "bridge_col_name": "State",
        "weak_label": None,
        "weak_score": None,
    }
    text_result = label_path(
        path,
        {
            "asset_type": "text",
            "content": "State: Texas",
            "entity_wiki_title": "Alpha",
        },
    )
    image_result = label_path(
        path,
        {
            "asset_type": "image",
            "entity_wiki_title": "Alpha",
            "file_name": "Alpha-Texas.jpg",
            "metadata": {
                "extmetadata": {
                    "ImageDescription": {"value": "Alpha in Texas"}
                }
            },
        },
    )

    assert text_result["weak_label"] == "weak_indirect"
    assert text_result["weak_score"] == 0.5
    assert image_result["weak_label"] is None
    assert image_result["weak_score"] is None


def test_stage1_evidence_paths_ignore_raw_table_asset_links(tmp_path):
    input_dir = tmp_path / "input"
    stage1_dir = tmp_path / "stage1"
    for artifact in (
        "source_tables",
        "bridge_assets",
        "evidence_recoveries",
        "table_asset_links",
    ):
        (input_dir / artifact).mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir()

    source_table = {
        "source_table_id": "source-1",
        "columns": [
            {"column_index": 0, "column_name": "Name"},
            {"column_index": 1, "column_name": "State"},
        ],
        "rows": [
            {
                "row_id": 0,
                "cells": [
                    {"column_index": 0, "text": "Alpha"},
                    {"column_index": 1, "text": "Texas"},
                ],
            }
        ],
    }
    assets = [
        {
            "asset_id": "recovered-asset",
            "asset_type": "text",
            "content": "Alpha has State Texas.",
            "entity_wiki_title": "SECRET_RECOVERY_TITLE",
        },
        {
            "asset_id": "raw-linked-asset",
            "asset_type": "text",
            "content": "Unrelated content.",
            "entity_wiki_title": "SECRET_LINK_TITLE",
        },
    ]
    recoveries = [
        {
            "source_table_id": "source-1",
            "source_row_id": 0,
            "recovered_attribute": {
                "column_index": 1,
                "column_name": "State",
                "value": "Texas",
            },
            "evidence": {
                "asset_id": "recovered-asset",
                "asset_type": "text",
            },
        }
    ]
    raw_links = [
        {
            "source_table_id": "source-1",
            "row_id": 0,
            "column_index": 0,
            "entity_id": "SECRET_ENTITY_ID",
            "entity_wiki_title": "SECRET_LINK_TITLE",
            "asset_ids": ["raw-linked-asset"],
        }
    ]
    artifact_records = {
        "source_tables": [source_table],
        "bridge_assets": assets,
        "evidence_recoveries": recoveries,
        "table_asset_links": raw_links,
    }
    manifest_artifacts = {}
    for artifact, records in artifact_records.items():
        relative_path = f"{artifact}/part-00000.jsonl"
        write_jsonl(input_dir / relative_path, records)
        manifest_artifacts[artifact] = {
            "shards": [{"path": relative_path, "records": len(records)}]
        }
    (input_dir / "dataset_manifest.json").write_text(
        json.dumps({"artifacts": manifest_artifacts}),
        encoding="utf-8",
    )
    write_jsonl(
        stage1_dir / "logic_fragments.jsonl",
        [
            {
                "fragment_id": "query-hidden",
                "role": "left_hidden",
                "split": "train",
                "chain_id": "chain-1",
                "source_table_id": "source-1",
                "source_column_indices": [0],
                "hidden_bridge_col": 1,
                "source_row_indices": [0],
            },
            {
                "fragment_id": "target",
                "role": "right_target",
                "chain_id": "chain-1",
                "rows": [
                    {"cells": [{"column_index": 0, "text": "Texas"}]}
                ],
            },
        ],
    )

    build_evidence_paths(
        argparse.Namespace(
            input_dir=str(input_dir),
            stage1_dir=str(stage1_dir),
            seed=13,
        )
    )

    paths = list(iter_jsonl(stage1_dir / "evidence_paths.jsonl"))
    assert len(paths) == 1
    assert paths[0]["asset_id"] == "recovered-asset"
    assert paths[0]["entity_text"] == "Alpha"
    assert paths[0]["reason"] == "candidate_path:intrinsic_recovery:Q_hidden->asset->T"
    assert "entity_id" not in paths[0]
    assert "SECRET" not in json.dumps(paths[0])


def test_qwen_encoder_wrapper_dummy_text_mock_normalized():
    encoder = Qwen3VLEmbeddingEncoder(mock=True)
    arr = encoder.encode_texts(["hello", "world"])
    assert arr.shape == (2, 32)
    assert np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-5)


def test_content_only_embedding_prompt_mode_uses_intrinsic_instructions():
    instructions = stage1_embeddings.embedding_instructions(argparse.Namespace(embedding_prompt_mode="content_only"))

    assert set(instructions) == {"table", "text", "image"}
    assert all("intrinsic" in instruction for instruction in instructions.values())
    assert "connect" not in instructions["table"].casefold()
    assert "hidden table attributes" not in instructions["text"].casefold()
    assert "multimodal table discovery" not in instructions["image"].casefold()


def test_connectivity_embedding_mode_does_not_assume_object_relationships():
    instructions = stage1_embeddings.embedding_instructions(
        argparse.Namespace(embedding_prompt_mode="connectivity")
    )

    assert all("intrinsic" in instruction for instruction in instructions.values())
    assert all("do not assume" in instruction.casefold() for instruction in instructions.values())


def test_embedding_prompt_mode_invalidates_incompatible_cache(tmp_path):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()

    assert not stage1_embeddings.prompt_cache_compatible(emb_dir, "connectivity")
    assert not stage1_embeddings.prompt_cache_compatible(emb_dir, "content_only")

    (emb_dir / "embedding_stats.json").write_text(
        json.dumps({"embedding_prompt_mode": "content_only"}),
        encoding="utf-8",
    )

    assert not stage1_embeddings.prompt_cache_compatible(emb_dir, "content_only")

    (emb_dir / "embedding_stats.json").write_text(
        json.dumps(
            {
                "embedding_prompt_mode": "content_only",
                "table_serialization_version": stage1_embeddings.TABLE_SERIALIZATION_VERSION,
            }
        ),
        encoding="utf-8",
    )

    assert stage1_embeddings.prompt_cache_compatible(emb_dir, "content_only")
    assert not stage1_embeddings.prompt_cache_compatible(emb_dir, "connectivity")


def test_table_embedding_uses_selected_prompt_mode(tmp_path):
    stage1_dir = tmp_path / "stage1"
    write_jsonl(
        stage1_dir / "logic_fragments.jsonl",
        [
            {
                "fragment_id": "f1",
                "page_title": "Players",
                "columns": [{"column_index": 0, "column_name": "Name"}],
                "rows": [{"cells": [{"column_index": 0, "column_name": "Name", "text": "Messi"}]}],
            }
        ],
    )
    seen: dict[str, object] = {}

    class RecordingEncoder:
        def encode_tables(self, texts, instruction=None):
            seen["instruction"] = instruction
            seen["texts"] = texts
            return np.ones((len(texts), 2), dtype="float32")

    stage1_embeddings.encode_table_fragments(
        argparse.Namespace(
            stage1_dir=str(stage1_dir),
            max_table_rows=5,
            force_recompute=False,
            embedding_prompt_mode="content_only",
        ),
        RecordingEncoder(),
        stage1_dir / "embeddings",
    )

    assert seen["instruction"] == stage1_embeddings.EMBEDDING_INSTRUCTIONS["content_only"]["table"]
    assert "Players" not in seen["texts"][0]
    assert "Messi" in seen["texts"][0]


def test_qwen_encoder_progress_bar_updates_by_item(monkeypatch):
    events: list[tuple[str, object]] = []

    class FakeTqdm:
        def __init__(self, *, total: int, desc: str, unit: str) -> None:
            events.append(("init", (total, desc, unit)))

        def update(self, amount: int) -> None:
            events.append(("update", amount))

        def close(self) -> None:
            events.append(("close", None))

    encoder = Qwen3VLEmbeddingEncoder(mock=True, batch_size=2, progress=True)
    encoder.mock = False
    encoder._encode_batch = lambda batch: np.ones((len(batch), 2), dtype="float32")  # type: ignore[method-assign]
    monkeypatch.setattr(qwen3_vl_embedding, "tqdm", FakeTqdm)

    arr = encoder.encode_texts(["one", "two", "three"])

    assert arr.shape == (3, 2)
    assert events[0] == ("init", (3, "Encoding text embeddings", "item"))
    assert [event for event in events if event[0] == "update"] == [("update", 2), ("update", 1)]
    assert events[-1] == ("close", None)


def test_large_embedding_image_is_resized_under_pixel_limit(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    source = tmp_path / "large.png"
    Image.new("RGB", (100, 80), color=(30, 90, 120)).save(source)

    resized, record = stage1_embeddings.ensure_image_within_pixel_limit(
        source,
        tmp_path / "cache",
        max_pixels=1000,
    )

    assert resized != source
    assert record is not None
    assert record["source_pixels"] == 8000
    with Image.open(resized) as image:
        assert image.width * image.height <= 1000
    assert source.exists()


def test_qwen_image_limit_error_is_parsed():
    exc = ValueError(
        "Image size (260546715 pixels) exceeds limit of 89500000 pixels, "
        "could be decompression bomb DOS attack."
    )
    assert image_limit_from_exception(exc) == 89500000


def test_qwen_retry_resize_uses_dynamic_limit(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    source = tmp_path / "large.jpg"
    Image.new("RGB", (100, 80), color=(30, 90, 120)).save(source)
    batch = [{"image": str(source), "text": "prompt"}]

    resized_batch = resize_batch_images_for_limit(batch, tmp_path / "cache", 1000)

    assert resized_batch[0]["image"] != str(source)
    with Image.open(resized_batch[0]["image"]) as image:
        assert image.width * image.height <= 1000


def test_wikipedia_extract_text_assets_are_chunked(tmp_path):
    content = (
        "Alpha was founded in 1990 and became known for bridge evidence. "
        "It later expanded into several regions with detailed public records.\n\n"
        "The second paragraph contains target facts and additional context. "
        "It should be stored as a separate text asset chunk for retrieval."
    )
    chunks = split_text_asset_content(content, max_chars=120, min_chars=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)

    class FakeWikipediaClient:
        api_failures = 0

        def get_page(self, wiki_title):
            return {
                "extract": content,
                "canonicalurl": f"https://example.test/wiki/{wiki_title}",
                "images": [],
            }

    entities = [{"entity_id": "ent_alpha", "wiki_title": "Alpha"}]
    writer = ShardedJsonlWriter(tmp_path / "bridge_assets", max_records_per_shard=10)
    with writer:
        entity_to_assets, api_failures, text_count, image_count = build_bridge_assets(
            entities,
            max_entities=None,
            max_images_per_entity=0,
            text_asset_chunk_chars=120,
            min_text_asset_chunk_chars=40,
            max_text_asset_chunks_per_entity=0,
            wikipedia_client=FakeWikipediaClient(),
            asset_writer=writer,
            flush_every_records=10,
        )

    records = [
        json.loads(line)
        for path in writer.paths()
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert api_failures == 0
    assert image_count == 0
    assert text_count == len(chunks)
    assert entity_to_assets["ent_alpha"] == [record["asset_id"] for record in records]
    assert {record["source"] for record in records} == {"wikipedia_extract_chunk"}
    assert all(record["asset_type"] == "text" for record in records)
    assert [record["text_chunk_index"] for record in records] == list(range(len(records)))
    assert {record["text_chunk_count"] for record in records} == {len(records)}


def test_wikipedia_client_get_pages_batches_titles_with_pipe(tmp_path):
    class FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "query": {
                    "normalized": [{"from": "Alpha_Page", "to": "Alpha Page"}],
                    "redirects": [{"from": "Beta", "to": "Beta Target"}],
                    "pages": [
                        {
                            "pageid": 1,
                            "title": "Alpha Page",
                            "extract": "Alpha extract",
                            "canonicalurl": "https://example.test/wiki/Alpha_Page",
                            "images": [],
                        },
                        {
                            "pageid": 2,
                            "title": "Beta Target",
                            "extract": "Beta extract",
                            "canonicalurl": "https://example.test/wiki/Beta_Target",
                            "images": [],
                        },
                    ],
                }
            }

    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.calls = []

        def get(self, url, params=None, timeout=30):
            self.calls.append((url, params))
            return FakeResponse()

    client = WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="test",
    )
    client.session = FakeSession()

    pages = client.get_pages(["Alpha_Page", "Beta"])

    assert len(client.session.calls) == 1
    assert client.session.calls[0][1]["titles"] == "Alpha Page|Beta"
    assert client.session.calls[0][1]["maxlag"] == 5
    assert pages["Alpha Page"]["title"] == "Alpha Page"
    assert pages["Beta"]["title"] == "Beta Target"
    assert client.get_page("Alpha Page")["extract"] == "Alpha extract"


def test_wikipedia_user_agent_defaults_to_environment(monkeypatch):
    env_user_agent = "MMDDDatasetBuilder/1.0 (mailto:mmdd@example.com)"
    monkeypatch.setenv("WIKIPEDIA_USER_AGENT", env_user_agent)

    table_args = mm_table_dataset.parse_args(["--input_dir", "in", "--output_dir", "out"])
    join_args = join_dataset.parse_args(["--input_dir", "in", "--output_dir", "out"])

    assert table_args.wikipedia_user_agent == env_user_agent
    assert join_args.wikipedia_user_agent == env_user_agent


def test_wikipedia_user_agent_has_usable_builtin_default(monkeypatch):
    monkeypatch.delenv("WIKIPEDIA_USER_AGENT", raising=False)

    table_args = mm_table_dataset.parse_args(["--input_dir", "in", "--output_dir", "out"])
    join_args = join_dataset.parse_args(["--input_dir", "in", "--output_dir", "out"])

    expected = "MMDD-EntiTables-DatasetBuilder/1.0 (mailto:cy.ouyang@zju.edu.cn)"
    assert mm_table_dataset.DEFAULT_WIKIPEDIA_USER_AGENT == expected
    assert table_args.wikipedia_user_agent == expected
    assert join_args.wikipedia_user_agent == expected


def test_wikipedia_user_agent_cli_overrides_environment(monkeypatch):
    monkeypatch.setenv("WIKIPEDIA_USER_AGENT", "MMDDDatasetBuilder/1.0 (mailto:env@example.com)")

    args = join_dataset.parse_args(
        [
            "--input_dir",
            "in",
            "--output_dir",
            "out",
            "--wikipedia_user_agent",
            "MMDDDatasetBuilder/1.0 (mailto:cli@example.com)",
        ]
    )

    assert args.wikipedia_user_agent == "MMDDDatasetBuilder/1.0 (mailto:cli@example.com)"


@pytest.mark.parametrize(
    "parser",
    [mm_table_dataset.parse_args, join_dataset.parse_args],
)
def test_media_cli_defaults_follow_wikimedia_policy(parser):
    args = parser(["--input_dir", "in", "--output_dir", "out"])

    assert args.media_download_workers == 2
    assert args.media_max_mbps == 24.0
    assert args.media_max_retries == 5
    assert args.media_retry_base_seconds == 5.0
    assert args.media_retry_max_seconds == 60.0
    assert args.media_chunk_bytes == 131072


@pytest.mark.parametrize(
    ("builder", "parser", "field", "value", "message"),
    [
        (
            mm_table_dataset,
            mm_table_dataset.parse_args,
            "media_download_workers",
            3,
            "media workers must be in the range 1..2",
        ),
        (
            join_dataset,
            join_dataset.parse_args,
            "media_download_workers",
            3,
            "media workers must be in the range 1..2",
        ),
        (
            mm_table_dataset,
            mm_table_dataset.parse_args,
            "media_max_mbps",
            25.1,
            r"media max_mbps must be in the range \(0, 25\]",
        ),
        (
            join_dataset,
            join_dataset.parse_args,
            "media_max_mbps",
            25.1,
            r"media max_mbps must be in the range \(0, 25\]",
        ),
    ],
)
def test_media_cli_invalid_policy_fails_before_network_client(
    tmp_path,
    monkeypatch,
    builder,
    parser,
    field,
    value,
    message,
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    args = parser(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(tmp_path / "output"),
        ]
    )
    setattr(args, field, value)

    def fail_if_constructed(**_kwargs):
        raise AssertionError("network client must not be constructed")

    monkeypatch.setattr(builder, "WikipediaClient", fail_if_constructed)

    with pytest.raises(ValueError, match=message):
        builder.build_dataset(args)


@pytest.mark.parametrize(
    ("builder", "parser"),
    [
        (mm_table_dataset, mm_table_dataset.parse_args),
        (join_dataset, join_dataset.parse_args),
    ],
)
def test_media_policy_is_wired_to_builder_client_and_manifest(
    tmp_path,
    monkeypatch,
    builder,
    parser,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    failure_path = output_dir / "media_download_failures.jsonl"
    failure_path.write_text('{"old": true}\n', encoding="utf-8")
    captured = {}
    media_summary = {"successful_downloads": 2, "failure_records": 1}

    class FakeWikipediaClient:
        api_failures = 0

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def media_summary(self):
            return media_summary

    monkeypatch.setattr(builder, "WikipediaClient", FakeWikipediaClient)
    argv = [
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--media_download_workers",
        "1",
        "--media_max_mbps",
        "12.5",
        "--media_max_retries",
        "2",
        "--media_retry_base_seconds",
        "1.5",
        "--media_retry_max_seconds",
        "9",
        "--media_chunk_bytes",
        "4096",
    ]
    if builder is join_dataset:
        argv.append("--no_model_progress")

    stats = builder.build_dataset(parser(argv))
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert captured["media_config"] == MediaPolicyConfig(
        workers=1,
        max_mbps=12.5,
        max_retries=2,
        retry_base_seconds=1.5,
        retry_max_seconds=9.0,
        chunk_bytes=4096,
    )
    assert captured["media_failure_recorder"].path == failure_path
    assert failure_path.read_text(encoding="utf-8") == ""
    assert manifest["wikimedia_media"] == media_summary
    assert (
        "api_failures is retained for backward compatibility; use manifest.wikimedia_media for media transfer counters"
        in stats["notes"]
    )


@pytest.mark.parametrize(
    ("builder", "parser"),
    [
        (mm_table_dataset, mm_table_dataset.parse_args),
        (join_dataset, join_dataset.parse_args),
    ],
)
def test_builder_uses_builtin_wikipedia_user_agent_without_placeholder_warning(
    tmp_path,
    monkeypatch,
    caplog,
    builder,
    parser,
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    captured = {}

    class FakeWikipediaClient:
        api_failures = 0

        def __init__(self, **kwargs):
            captured["user_agent"] = kwargs["user_agent"]

        def media_summary(self):
            return {}

    monkeypatch.delenv("WIKIPEDIA_USER_AGENT", raising=False)
    monkeypatch.setattr(builder, "WikipediaClient", FakeWikipediaClient)
    argv = [
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(tmp_path / "output"),
    ]
    if builder is join_dataset:
        argv.append("--no_model_progress")

    with caplog.at_level("WARNING"):
        builder.build_dataset(parser(argv))

    assert captured["user_agent"] == mm_table_dataset.DEFAULT_WIKIPEDIA_USER_AGENT
    assert "placeholder" not in caplog.text.lower()


def test_wikipedia_client_limits_action_api_to_five_requests_per_second(tmp_path, monkeypatch):
    now = [100.0]
    sleeps = []

    class FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"query": {"pages": []}}

    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.calls = 0

        def get(self, url, params=None, timeout=30):
            self.calls += 1
            return FakeResponse()

    monkeypatch.setattr(mm_table_dataset, "_ACTION_API_LAST_REQUEST_TIME", 0.0, raising=False)
    monkeypatch.setattr(mm_table_dataset.time, "monotonic", lambda: now[0])

    def fake_sleep(seconds):
        sleeps.append(round(seconds, 6))
        now[0] += seconds

    monkeypatch.setattr(mm_table_dataset.time, "sleep", fake_sleep)

    client = WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
    )
    client.session = FakeSession()

    assert client._get({"action": "query"}) == {"query": {"pages": []}}
    assert client._get({"action": "query"}) == {"query": {"pages": []}}

    assert client.session.calls == 2
    assert sleeps == [0.2]


def test_wikipedia_action_api_rate_limit_is_shared_across_clients(tmp_path, monkeypatch):
    now = [200.0]
    sleeps = []

    class FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"query": {"pages": []}}

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, params=None, timeout=30):
            return FakeResponse()

    monkeypatch.setattr(mm_table_dataset, "_ACTION_API_LAST_REQUEST_TIME", 0.0, raising=False)
    monkeypatch.setattr(mm_table_dataset.time, "monotonic", lambda: now[0])

    def fake_sleep(seconds):
        sleeps.append(round(seconds, 6))
        now[0] += seconds

    monkeypatch.setattr(mm_table_dataset.time, "sleep", fake_sleep)

    client_a = WikipediaClient(
        cache_dir=tmp_path / "cache_a",
        image_output_dir=tmp_path / "images_a",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
    )
    client_b = WikipediaClient(
        cache_dir=tmp_path / "cache_b",
        image_output_dir=tmp_path / "images_b",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
    )
    client_a.session = FakeSession()
    client_b.session = FakeSession()

    assert client_a._get({"action": "query"}) == {"query": {"pages": []}}
    assert client_b._get({"action": "query"}) == {"query": {"pages": []}}

    assert sleeps == [0.2]


def test_wikipedia_client_retries_rate_limit_with_retry_after(tmp_path, monkeypatch):
    sleeps = []

    class FakeResponse:
        headers = {}

        def __init__(self, status_code, payload=None, headers=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.calls = 0

        def get(self, url, params=None, timeout=30):
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(429, headers={"Retry-After": "3"})
            return FakeResponse(200, {"query": {"pages": []}})

    monkeypatch.setattr(mm_table_dataset.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(mm_table_dataset, "_ACTION_API_LAST_REQUEST_TIME", 0.0, raising=False)

    client = WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
    )
    client.session = FakeSession()

    assert client._get({"action": "query"}) == {"query": {"pages": []}}
    assert client.session.calls == 2
    assert sleeps[0] == 3.0
    assert client.api_failures == 0


def test_wikipedia_client_uses_descriptive_user_agent(tmp_path):
    client = WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
    )

    assert "DatasetBot/1.0" in client.session.headers["User-Agent"]
    assert client.session.headers["Accept-Encoding"] == "gzip, deflate"


def test_wikipedia_client_exposes_media_summary(tmp_path):
    client = WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
    )
    client.media_downloader.stats.increment("retry_attempts", 2)

    assert client.media_summary()["retry_attempts"] == 2


def test_build_bridge_assets_fetches_wikipedia_pages_in_batches(tmp_path):
    class FakeWikipediaClient:
        api_failures = 0

        def __init__(self):
            self.page_batches = []
            self.imageinfo_batches = []

        def get_pages(self, wiki_titles):
            titles = list(wiki_titles)
            self.page_batches.append(titles)
            return {
                title: {
                    "wiki_title": title,
                    "title": title,
                    "extract": f"{title} extract",
                    "canonicalurl": f"https://example.test/wiki/{title}",
                    "images": [],
                }
                for title in titles
            }

        def get_imageinfos(self, file_titles):
            self.imageinfo_batches.append(list(file_titles))
            return {}

    client = FakeWikipediaClient()
    writer = ShardedJsonlWriter(tmp_path / "bridge_assets", max_records_per_shard=10)
    entities = [
        {"entity_id": "ent_alpha", "wiki_title": "Alpha", "display_texts": ["Alpha"]},
        {"entity_id": "ent_beta", "wiki_title": "Beta", "display_texts": ["Beta"]},
    ]

    with writer:
        entity_to_assets, api_failures, text_count, image_count = build_bridge_assets(
            entities,
            max_entities=None,
            max_images_per_entity=0,
            text_asset_chunk_chars=120,
            min_text_asset_chunk_chars=10,
            max_text_asset_chunks_per_entity=1,
            wikipedia_client=client,
            asset_writer=writer,
            flush_every_records=10,
        )

    assert client.page_batches == [["Alpha", "Beta"]]
    assert client.imageinfo_batches == []
    assert api_failures == 0
    assert text_count == 2
    assert image_count == 0
    assert sorted(entity_to_assets) == ["ent_alpha", "ent_beta"]


def test_wikipedia_client_imageinfo_requests_thumbnail_url_and_caches_fields(tmp_path):
    class FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "query": {
                    "pages": [
                        {
                            "pageid": 10,
                            "title": "File:Alpha.jpg",
                            "imageinfo": [
                                {
                                    "url": "https://upload.wikimedia.org/original/Alpha.jpg",
                                    "thumburl": "https://upload.wikimedia.org/thumb/Alpha.jpg/960px-Alpha.jpg",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Alpha.jpg",
                                    "mime": "image/jpeg",
                                    "mediatype": "BITMAP",
                                    "width": 1200,
                                    "height": 800,
                                    "thumbwidth": 960,
                                    "thumbheight": 640,
                                    "size": 12345,
                                    "extmetadata": {"Artist": {"value": "Someone"}},
                                }
                            ],
                        }
                    ]
                }
            }

    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.calls = []

        def get(self, url, params=None, timeout=30):
            self.calls.append((url, params))
            return FakeResponse()

    client = WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
    )
    client.session = FakeSession()

    imageinfos = client.get_imageinfos(["File:Alpha.jpg"])

    assert client.session.calls[0][1]["iiurlwidth"] == 960
    assert client.session.calls[0][1]["maxlag"] == 5
    assert imageinfos["File:Alpha.jpg"]["thumburl"].endswith("960px-Alpha.jpg")
    assert imageinfos["File:Alpha.jpg"]["thumbwidth"] == 960
    assert imageinfos["File:Alpha.jpg"]["thumbheight"] == 640


def jpeg_info(name):
    return {
        "file_title": f"File:{name}",
        "url": f"https://upload.wikimedia.org/original/{name}",
        "thumburl": f"https://upload.wikimedia.org/thumb/{name}/960px-{name}",
        "mime": "image/jpeg",
        "mediatype": "BITMAP",
        "width": 1200,
        "height": 800,
        "thumbwidth": 960,
        "thumbheight": 640,
    }


class SessionThatFailsOnGet:
    headers = {}

    def get(self, *args, **kwargs):
        raise AssertionError("cache hit must not access the network")


class FakeStreamingResponse:
    def __init__(self, status_code, body=b"", headers=None, chunks=None):
        self.status_code = status_code
        self.body = body
        self.headers = dict(headers or {})
        self.text = body.decode("utf-8", errors="replace")
        self.chunks = list(chunks) if chunks is not None else [body]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def iter_content(self, chunk_size):
        yield from self.chunks


class SequenceImageSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.calls = 0
        self._lock = threading.Lock()

    def get(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
            return self.responses.pop(0)


def make_wikipedia_client(tmp_path, **media_overrides):
    recorder = MediaFailureRecorder(tmp_path / "failures.jsonl")
    recorder.reset()
    config = {
        "max_retries": 0,
        "retry_base_seconds": 0,
        "retry_max_seconds": 0,
    }
    config.update(media_overrides)
    return WikipediaClient(
        cache_dir=tmp_path / "metadata",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
        media_config=MediaPolicyConfig(**config),
        media_failure_recorder=recorder,
    )


def test_download_image_reuses_legacy_asset_file(tmp_path):
    client = make_wikipedia_client(tmp_path)
    legacy = tmp_path / "images" / "asset_alpha.jpg"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    client.session = SessionThatFailsOnGet()

    record = client.download_image(jpeg_info("Alpha.jpg"), "asset_alpha")

    assert Path(record["local_path"]) == legacy
    assert record["downloaded"] is False
    assert client.media_downloader.summary()["legacy_cache_hits"] == 1


def test_download_image_reuses_url_keyed_file_for_distinct_assets(tmp_path):
    client = make_wikipedia_client(tmp_path)
    session = SequenceImageSession(
        [FakeStreamingResponse(200, b"image", {"Content-Type": "image/jpeg"})]
    )
    client.session = session

    first = client.download_image(jpeg_info("Shared.jpg"), "asset_a")
    second = client.download_image(jpeg_info("Shared.jpg"), "asset_b")

    assert first["local_path"] == second["local_path"]
    assert Path(first["local_path"]).name.startswith("media_")
    assert first["downloaded"] is True
    assert second["downloaded"] is False
    assert session.calls == 1
    assert client.media_downloader.summary()["shared_cache_hits"] == 1


def test_download_image_singleflight_shares_one_network_attempt(tmp_path):
    started = threading.Event()
    release = threading.Event()
    callers_ready = threading.Barrier(2)

    class BlockingResponse(FakeStreamingResponse):
        def iter_content(self, chunk_size):
            started.set()
            assert release.wait(timeout=2)
            yield self.body

    client = make_wikipedia_client(tmp_path)
    session = SequenceImageSession(
        [BlockingResponse(200, b"image", {"Content-Type": "image/jpeg"})]
    )
    client.session = session
    original_shared_media_path = client._shared_media_path

    def synchronized_shared_media_path(url, extension):
        path = original_shared_media_path(url, extension)
        callers_ready.wait(timeout=2)
        return path

    client._shared_media_path = synchronized_shared_media_path

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.download_image, jpeg_info("Shared.jpg"), "asset_a")
        second = pool.submit(client.download_image, jpeg_info("Shared.jpg"), "asset_b")
        assert started.wait(timeout=2)
        release.set()
        records = [first.result(timeout=2), second.result(timeout=2)]

    assert records[0] == records[1]
    assert session.calls == 1
    assert client.media_downloader.summary()["singleflight_followers"] == 1


@pytest.mark.parametrize(
    ("responses", "expected_calls", "expect_success"),
    [
        (
            [
                FakeStreamingResponse(429, b"slow down", {"Retry-After": "0"}),
                FakeStreamingResponse(200, b"image", {"Content-Type": "image/jpeg"}),
            ],
            2,
            True,
        ),
        (
            [
                FakeStreamingResponse(503, b"unavailable"),
                FakeStreamingResponse(200, b"image", {"Content-Type": "image/jpeg"}),
            ],
            2,
            True,
        ),
        ([FakeStreamingResponse(404, b"missing")], 1, False),
        (
            [
                FakeStreamingResponse(
                    429,
                    b"Use thumbnail steps",
                    {"Retry-After": "0"},
                )
            ],
            1,
            False,
        ),
    ],
)
def test_download_image_coordinates_retryable_and_terminal_http_responses(
    tmp_path,
    responses,
    expected_calls,
    expect_success,
):
    client = make_wikipedia_client(tmp_path, max_retries=1)
    session = SequenceImageSession(responses)
    client.session = session

    record = client.download_image(jpeg_info("Retry.jpg"), "asset_retry")

    assert (record is not None) is expect_success
    assert session.calls == expected_calls
    assert not list((tmp_path / "images").glob("*.tmp"))


def test_download_image_rejects_non_image_and_removes_temporary_file(tmp_path):
    client = make_wikipedia_client(tmp_path)
    client.session = SequenceImageSession(
        [FakeStreamingResponse(200, b"<html>no</html>", {"Content-Type": "text/html"})]
    )

    record = client.download_image(jpeg_info("NotImage.jpg"), "asset_html")

    assert record is None
    assert not list((tmp_path / "images").glob("*.tmp"))
    failure = json.loads((tmp_path / "failures.jsonl").read_text().strip())
    assert failure["exception_class"] == NonImageMediaError.__name__


def test_empty_200_response_is_terminal_for_current_run_and_retryable_next_run(
    tmp_path,
):
    info = jpeg_info("Empty.jpg")
    client = make_wikipedia_client(tmp_path)
    session = SequenceImageSession(
        [
            FakeStreamingResponse(
                200,
                b"",
                {"Content-Type": "image/jpeg"},
                chunks=[b"", b""],
            )
        ]
    )
    client.session = session
    shared_path = client._shared_media_path(info["thumburl"], ".jpg")

    first = client.download_image(info, "asset_empty_a")
    repeated = client.download_image(info, "asset_empty_b")

    assert first is None
    assert repeated is None
    assert session.calls == 1
    assert not shared_path.exists()
    assert not list((tmp_path / "images").glob("*.tmp"))
    failure = json.loads((tmp_path / "failures.jsonl").read_text().strip())
    assert failure["exception_class"] == NonImageMediaError.__name__
    assert client.media_downloader.summary().get("successful_downloads", 0) == 0

    next_run = make_wikipedia_client(tmp_path)
    next_session = SequenceImageSession(
        [FakeStreamingResponse(200, b"image", {"Content-Type": "image/jpeg"})]
    )
    next_run.session = next_session

    retried = next_run.download_image(info, "asset_empty_c")

    assert retried is not None
    assert Path(retried["local_path"]).read_bytes() == b"image"
    assert next_session.calls == 1


def test_download_attempt_reserves_before_read_and_refunds_short_chunk(tmp_path):
    events = []

    class OrderedResponse(FakeStreamingResponse):
        def iter_content(self, chunk_size):
            events.append(("read", len(self.body)))
            yield self.body

    class RecordingLimiter:
        def __init__(self):
            self.acquires = 0

        def acquire(self, byte_count):
            self.acquires += 1
            events.append(("acquire", byte_count))
            return 0.25 if self.acquires == 1 else 0.0

        def refund(self, byte_count):
            events.append(("refund", byte_count))

    client = make_wikipedia_client(tmp_path, chunk_bytes=16)
    client.session = SequenceImageSession(
        [OrderedResponse(200, b"short", {"Content-Type": "image/jpeg"})]
    )
    image_path = client._shared_media_path(
        jpeg_info("Ordered.jpg")["thumburl"],
        ".jpg",
    )
    image_path.parent.mkdir(parents=True)

    record = client._download_image_attempt(
        jpeg_info("Ordered.jpg"),
        jpeg_info("Ordered.jpg")["thumburl"],
        "thumbnail",
        image_path,
        RecordingLimiter(),
    )

    assert events.index(("acquire", 16)) < events.index(("read", 5))
    assert ("refund", 11) in events
    assert record["bytes"] == 5
    assert client.media_downloader.summary()["bandwidth_wait_seconds"] == 0.25


def test_download_attempt_throttles_http_error_body_before_classification(tmp_path):
    events = []

    class ErrorResponse:
        status_code = 429
        headers = {"Retry-After": "0"}
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.closed = True
            return False

        @property
        def text(self):
            raise AssertionError("error body must be streamed through the limiter")

        def iter_content(self, chunk_size):
            events.append(("chunk_size", chunk_size))
            events.append(("read", len(b"slow down")))
            yield b"slow down"
            for _ in range(1000):
                events.append(("read", chunk_size))
                yield b"x" * chunk_size

    class RecordingLimiter:
        def acquire(self, byte_count):
            events.append(("acquire", byte_count))
            return 0.0

        def refund(self, byte_count):
            events.append(("refund", byte_count))

    response = ErrorResponse()
    client = make_wikipedia_client(tmp_path)
    client.session = SequenceImageSession([response])
    info = jpeg_info("RateLimited.jpg")

    with pytest.raises(mm_table_dataset.MediaHTTPError) as error:
        client._download_image_attempt(
            info,
            info["thumburl"],
            "thumbnail",
            client._shared_media_path(info["thumburl"], ".jpg"),
            RecordingLimiter(),
        )

    assert error.value.status_code == 429
    assert error.value.body_excerpt == "slow down"
    assert events == [
        ("acquire", 4096),
        ("chunk_size", 4096),
        ("read", 9),
        ("refund", 4087),
    ]
    assert response.closed is True
    assert not list((tmp_path / "images").glob("*.tmp"))


def test_download_attempt_promotes_complete_file_atomically(tmp_path):
    client = make_wikipedia_client(tmp_path)
    info = jpeg_info("Atomic.jpg")
    image_path = client._shared_media_path(info["thumburl"], ".jpg")
    tmp_pathname = image_path.with_suffix(image_path.suffix + ".tmp")

    class AtomicResponse(FakeStreamingResponse):
        def iter_content(self, chunk_size):
            assert tmp_pathname.exists()
            assert not image_path.exists()
            yield self.body

    client.session = SequenceImageSession(
        [AtomicResponse(200, b"complete", {"Content-Type": "image/jpeg"})]
    )

    record = client.download_image(info, "asset_atomic")

    assert Path(record["local_path"]) == image_path
    assert image_path.read_bytes() == b"complete"
    assert not tmp_pathname.exists()


def test_wikipedia_download_image_prefers_safe_thumbnail_url(tmp_path):
    image_bytes = b"fake-jpeg-bytes"

    class FakeResponse:
        status_code = 200
        headers = {"Content-Type": "image/jpeg"}
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield image_bytes

    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.urls = []

        def get(self, url, stream=True, timeout=60):
            self.urls.append(url)
            return FakeResponse()

    client = WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
    )
    client.session = FakeSession()

    record = client.download_image(
        {
            "url": "https://upload.wikimedia.org/original/Alpha.jpg",
            "thumburl": "https://upload.wikimedia.org/thumb/Alpha.jpg/768px-Alpha.jpg",
            "mime": "image/jpeg",
            "width": 1200,
            "height": 800,
            "thumbwidth": 768,
            "thumbheight": 512,
        },
        "asset_alpha",
    )

    assert record is not None
    assert client.session.urls == ["https://upload.wikimedia.org/thumb/Alpha.jpg/768px-Alpha.jpg"]
    assert record["download_url"] == "https://upload.wikimedia.org/thumb/Alpha.jpg/768px-Alpha.jpg"
    assert record["download_kind"] == "thumbnail"


def test_wikipedia_download_image_avoids_upscaled_raster_thumbnail(tmp_path):
    image_bytes = b"fake-jpeg-bytes"

    class FakeResponse:
        status_code = 200
        headers = {"Content-Type": "image/jpeg"}
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield image_bytes

    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.urls = []

        def get(self, url, stream=True, timeout=60):
            self.urls.append(url)
            return FakeResponse()

    client = WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
    )
    client.session = FakeSession()

    record = client.download_image(
        {
            "url": "https://upload.wikimedia.org/original/Small.jpg",
            "thumburl": "https://upload.wikimedia.org/thumb/Small.jpg/768px-Small.jpg",
            "mime": "image/jpeg",
            "width": 320,
            "height": 200,
            "thumbwidth": 768,
            "thumbheight": 480,
        },
        "asset_small",
    )

    assert record is not None
    assert client.session.urls == ["https://upload.wikimedia.org/original/Small.jpg"]
    assert record["download_kind"] == "original"


def test_build_bridge_assets_does_not_block_imageinfo_on_prior_image_download(tmp_path, monkeypatch):
    events = []
    progress_events = []
    beta_imageinfo_seen = threading.Event()

    def fake_tqdm(iterable, **kwargs):
        progress_events.append(kwargs)
        return iterable

    monkeypatch.setattr(mm_table_dataset, "tqdm", fake_tqdm)

    class FakeWikipediaClient:
        api_failures = 0

        def get_pages(self, wiki_titles):
            return {
                "Alpha": {
                    "extract": "Alpha extract",
                    "canonicalurl": "https://example.test/wiki/Alpha",
                    "images": [{"title": "File:Alpha.jpg"}],
                },
                "Beta": {
                    "extract": "Beta extract",
                    "canonicalurl": "https://example.test/wiki/Beta",
                    "images": [{"title": "File:Beta.jpg"}],
                },
            }

        def get_imageinfos(self, file_titles):
            titles = list(file_titles)
            events.append(("imageinfos", tuple(titles)))
            if "File:Beta.jpg" in titles:
                beta_imageinfo_seen.set()
            return {
                title: {
                    "file_title": title,
                    "url": f"https://upload.wikimedia.org/original/{title.removeprefix('File:')}",
                    "mime": "image/jpeg",
                    "mediatype": "BITMAP",
                    "width": 1200,
                    "height": 800,
                    "size": 100,
                    "extmetadata": {},
                }
                for title in titles
            }

        def download_image(self, imageinfo, asset_id):
            file_title = imageinfo["file_title"]
            events.append(("download_start", file_title))
            if file_title == "File:Alpha.jpg":
                beta_imageinfo_seen.wait(timeout=0.5)
            events.append(("download_done", file_title))
            path = tmp_path / f"{asset_id}.jpg"
            path.write_bytes(b"image")
            return {
                "local_path": str(path),
                "relative_path": path.name,
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": "sha",
                "downloaded": True,
            }

    client = FakeWikipediaClient()
    writer = ShardedJsonlWriter(tmp_path / "bridge_assets", max_records_per_shard=20)
    entities = [
        {"entity_id": "ent_alpha", "wiki_title": "Alpha", "display_texts": ["Alpha"]},
        {"entity_id": "ent_beta", "wiki_title": "Beta", "display_texts": ["Beta"]},
    ]

    with writer:
        _entity_to_assets, _api_failures, _text_count, image_count = build_bridge_assets(
            entities,
            max_entities=None,
            max_images_per_entity=1,
            text_asset_chunk_chars=120,
            min_text_asset_chunk_chars=10,
            max_text_asset_chunks_per_entity=1,
            wikipedia_client=client,
            asset_writer=writer,
            flush_every_records=20,
        )

    assert image_count == 2
    assert events.index(("imageinfos", ("File:Beta.jpg",))) < events.index(("download_done", "File:Alpha.jpg"))
    assert {
        "total": 2,
        "desc": "Resolving Wikimedia media",
        "unit": "asset",
        "dynamic_ncols": True,
    } in progress_events


def test_joinability_wikipedia_workers_report_serial_without_parallel_builder(tmp_path, monkeypatch):
    parallel_calls = 0
    output_dir = tmp_path / "out"

    monkeypatch.setattr(join_dataset, "read_entitables_json", lambda _input_dir: [])
    monkeypatch.setattr(join_dataset, "parse_source_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        join_dataset,
        "write_sharded_jsonl",
        lambda output_path, *_args, **_kwargs: ShardedJsonlWriter(output_path, 10),
    )

    def fake_build_bridge_assets_parallel(*_args, **_kwargs):
        nonlocal parallel_calls
        parallel_calls += 1
        return {}, 0, 0, 0

    monkeypatch.setattr(join_dataset, "build_bridge_assets_parallel", fake_build_bridge_assets_parallel)

    class FakeWikipediaClient:
        api_failures = 0

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(join_dataset, "WikipediaClient", FakeWikipediaClient)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_mm_joinability_dataset.py",
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(output_dir),
            "--wikipedia_workers",
            "4",
        ],
    )

    join_dataset.main()

    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    assert parallel_calls == 0
    assert stats["wikipedia_workers"] == 1
    assert manifest["wikipedia_cache"]["workers"] == 1


def test_text_asset_chunk_limit_keeps_relevant_chunk(tmp_path):
    content = (
        "Alpha has a long public biography with unrelated background details.\n\n"
        "Alpha represented Arkansas and this paragraph contains the bridge evidence.\n\n"
        "Alpha also appeared in other unrelated references."
    )

    class FakeWikipediaClient:
        api_failures = 0

        def get_page(self, wiki_title):
            return {
                "extract": content,
                "canonicalurl": f"https://example.test/wiki/{wiki_title}",
                "images": [],
            }

    entities = [
        {
            "entity_id": "ent_alpha",
            "wiki_title": "Alpha",
            "display_texts": ["Alpha"],
            "context_terms": ["Arkansas", "State"],
        }
    ]
    writer = ShardedJsonlWriter(tmp_path / "bridge_assets", max_records_per_shard=10)
    with writer:
        entity_to_assets, _api_failures, text_count, image_count = build_bridge_assets(
            entities,
            max_entities=None,
            max_images_per_entity=0,
            text_asset_chunk_chars=120,
            min_text_asset_chunk_chars=20,
            max_text_asset_chunks_per_entity=1,
            wikipedia_client=FakeWikipediaClient(),
            asset_writer=writer,
            flush_every_records=10,
        )

    records = [
        json.loads(line)
        for path in writer.paths()
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert image_count == 0
    assert text_count == 1
    assert len(entity_to_assets["ent_alpha"]) == 1
    assert records[0]["content"] == "Alpha represented Arkansas and this paragraph contains the bridge evidence."
    assert records[0]["text_chunk_index"] == 1
    assert records[0]["selected_text_chunk_count"] == 1
    assert records[0]["text_chunk_relevance_score"] > 0


def read_manifest_artifact(root: Path, artifact: str) -> list[dict[str, object]]:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    records = []
    for shard in manifest["artifacts"][artifact]["shards"]:
        records.extend(json.loads(line) for line in (root / shard["path"]).read_text(encoding="utf-8").splitlines())
    return records


def test_joinability_dataset_keeps_note_like_columns_as_candidates_and_context():
    table = {
        "columns": [
            {"column_index": 0, "column_name": "Entity"},
            {"column_index": 1, "column_name": "City"},
            {"column_index": 2, "column_name": "Role"},
            {"column_index": 3, "column_name": "Notes"},
            {"column_index": 4, "column_name": "Source"},
        ],
        "metadata": {
            "column_profiles": [
                {"column_index": idx, "non_empty_ratio": 1.0, "unique_ratio": 0.5}
                for idx in range(5)
            ]
        },
    }

    assert join_dataset.candidate_attribute_columns(table, entity_col=0, min_non_empty_ratio=0.5) == [1, 2, 3, 4]
    assert set(join_dataset.context_columns(table, excluded={0, 1}, limit=0)) == {2, 3, 4}


def test_joinability_dataset_accepts_external_wikipedia_cache_dirs(tmp_path, monkeypatch):
    input_dir = tmp_path / "entitables"
    output_dir = tmp_path / "joinability"
    old_cache_dir = tmp_path / "previous_joinability" / "cache"
    old_image_dir = tmp_path / "previous_joinability" / "images"
    input_dir.mkdir()
    output_dir.mkdir()
    failure_ledger = output_dir / "media_download_failures.jsonl"
    failure_ledger.write_text('{"old": true}\n', encoding="utf-8")
    (input_dir / "tables.json").write_text("{}", encoding="utf-8")
    captured: dict[str, Path] = {}

    class FakeWikipediaClient:
        api_failures = 0

        def __init__(
            self,
            cache_dir,
            image_output_dir,
            output_dir,
            sleep,
            user_agent,
            *,
            media_config,
            media_failure_recorder,
        ):
            captured["cache_dir"] = Path(cache_dir)
            captured["image_output_dir"] = Path(image_output_dir)
            captured["output_dir"] = Path(output_dir)

        def get_page(self, wiki_title):
            raise AssertionError("empty input should not request Wikipedia pages")

    monkeypatch.setattr(join_dataset, "WikipediaClient", FakeWikipediaClient)
    args = join_dataset.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--cache_dir",
            str(tmp_path / "shared_cache"),
            "--wikipedia_cache_dir",
            str(old_cache_dir),
            "--wikipedia_image_dir",
            str(old_image_dir),
            "--no_model_progress",
        ]
    )

    stats = join_dataset.build_dataset(args)
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert stats["source_tables"] == 0
    assert captured["cache_dir"] == old_cache_dir.resolve()
    assert captured["image_output_dir"] == old_image_dir.resolve()
    assert captured["output_dir"] == output_dir.resolve()
    assert failure_ledger.read_text(encoding="utf-8") == ""
    assert manifest["wikipedia_cache"]["cache_dir"] == str(old_cache_dir.resolve())
    assert manifest["wikipedia_cache"]["image_dir"] == str(old_image_dir.resolve())


def test_table_builder_resets_media_failure_ledger_before_wikipedia_work(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "entitables"
    output_dir = tmp_path / "table_dataset"
    input_dir.mkdir()
    output_dir.mkdir()
    failure_ledger = output_dir / "media_download_failures.jsonl"
    failure_ledger.write_text('{"old": true}\n', encoding="utf-8")

    class FakeWikipediaClient:
        api_failures = 0

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(mm_table_dataset, "WikipediaClient", FakeWikipediaClient)
    args = mm_table_dataset.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
        ]
    )

    mm_table_dataset.build_dataset(args)

    assert failure_ledger.read_text(encoding="utf-8") == ""


def test_joinability_dataset_uses_shared_cache_dir_by_default(tmp_path, monkeypatch):
    input_dir = tmp_path / "entitables"
    output_dir = tmp_path / "joinability"
    input_dir.mkdir()
    (input_dir / "tables.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    captured: dict[str, Path] = {}

    class FakeWikipediaClient:
        api_failures = 0

        def __init__(
            self,
            cache_dir,
            image_output_dir,
            output_dir,
            sleep,
            user_agent,
            *,
            media_config,
            media_failure_recorder,
        ):
            captured["cache_dir"] = Path(cache_dir)
            captured["image_output_dir"] = Path(image_output_dir)

        def get_page(self, wiki_title):
            raise AssertionError("empty input should not request Wikipedia pages")

    monkeypatch.setattr(join_dataset, "WikipediaClient", FakeWikipediaClient)
    args = join_dataset.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--no_model_progress",
        ]
    )

    join_dataset.build_dataset(args)
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    shared_cache_dir = (tmp_path / "cache" / "mm_joinability").resolve()

    assert captured["cache_dir"] == shared_cache_dir / "wikipedia"
    assert captured["image_output_dir"] == shared_cache_dir / "images"
    assert manifest["cache"]["root_dir"] == str(shared_cache_dir)
    assert manifest["cache"]["model_attribute_extractions"] == str(shared_cache_dir / "model_attribute_extractions.jsonl")
    assert not (output_dir / "cache").exists()


def test_joinability_dataset_maps_evidence_to_query_entity_attribute(tmp_path, monkeypatch):
    input_dir = tmp_path / "entitables"
    input_dir.mkdir()
    output_dir = tmp_path / "joinability"
    queryable_table = {
        "title": ["Entity", "City", "Team", "League"],
        "numCols": 4,
        "numericColumns": [],
        "pgTitle": "Queryable Page",
        "numDataRows": 6,
        "secondTitle": "Section",
        "caption": "Caption",
        "data": [
            ["[Alpha_Page|Alpha]", "Paris", "Red", "One"],
            ["[Delta_Page|Delta]", "", "Green", "Two"],
            ["[Beta_Page|Beta]", "Paris", "Red", "One"],
            ["[Epsilon_Page|Epsilon]", "", "Yellow", "Three"],
            ["[Gamma_Page|Gamma]", "Oslo", "Blue", "Four"],
            ["[Zeta_Page|Zeta]", "Madrid", "Black", "Five"],
        ],
    }
    rejected_table = {
        "title": ["Entity", "City", "Team", "League"],
        "numCols": 4,
        "numericColumns": [],
        "pgTitle": "Rejected Page",
        "numDataRows": 5,
        "secondTitle": "Section",
        "caption": "Caption",
        "data": [
            ["[No_A|No A]", "Madrid", "One", "A"],
            ["[No_B|No B]", "Berlin", "Two", "B"],
            ["[No_C|No C]", "Lisbon", "Three", "C"],
            ["[No_D|No D]", "Dublin", "Four", "D"],
            ["[No_E|No E]", "Rome", "Five", "E"],
        ],
    }
    (input_dir / "tables.json").write_text(json.dumps({"table_1": queryable_table, "table_2": rejected_table}), encoding="utf-8")

    class FakeWikipediaClient:
        api_failures = 0

        def __init__(self, *args, **kwargs):
            pass

        def get_page(self, wiki_title):
            city_by_title = {
                "Alpha Page": "Paris",
                "Beta Page": "Paris",
                "Gamma Page": "Oslo",
            }
            city = city_by_title.get(wiki_title, "")
            return {
                "extract": f"{wiki_title} biography. City: {city}." if city else f"{wiki_title} biography without recoverable table attributes.",
                "canonicalurl": f"https://example.test/wiki/{wiki_title}",
                "images": [],
            }

    class FakeAttributeExtractor:
        def __init__(self, args):
            pass

        def extract(self, asset, entity, candidate_attribute_names):
            content = asset.get("content", "")
            attrs = []
            for city in ("Paris", "Oslo", "Madrid", "Berlin", "Lisbon", "Dublin"):
                if city in content:
                    attrs.append(
                        {
                            "name": "City",
                            "value": city,
                            "evidence": f"City: {city}",
                            "connection_evidence": f"{entity['cell_text']} is the subject of this Wikipedia article.",
                        }
                    )
            return {"attributes": attrs, "raw_response": json.dumps({"attributes": attrs}), "error": ""}

    monkeypatch.setattr(join_dataset, "WikipediaClient", FakeWikipediaClient)
    monkeypatch.setattr(join_dataset, "LocalAttributeExtractor", FakeAttributeExtractor)
    args = join_dataset.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--cache_dir",
            str(tmp_path / "shared_cache"),
            "--max_source_tables",
            "2",
            "--max_entities",
            "20",
            "--max_images_per_entity",
            "0",
            "--text_asset_chunk_chars",
            "300",
            "--min_text_asset_chunk_chars",
            "20",
            "--max_text_asset_chunks_per_entity",
            "1",
            "--wiki_link_threshold",
            "0.5",
            "--min_rows",
            "4",
            "--min_cols",
            "3",
            "--min_rows_per_output_table",
            "2",
            "--query_rows_per_table",
            "5",
            "--min_recovered_value_ratio",
            "0.6",
            "--min_recovery_denominator",
            "4",
            "--max_query_context_attrs",
            "1",
            "--max_target_context_attrs",
            "1",
            "--sleep",
            "0",
            "--records_per_shard",
            "10",
            "--flush_every_records",
            "2",
            "--split_by",
            "source_table_id",
        ]
    )

    stats = join_dataset.build_dataset(args)

    assert stats["query_tables"] == 1
    assert stats["data_lake_tables"] == 2
    assert stats["queryable_source_tables"] == 1
    assert stats["rejected_source_tables"] == 1
    assert stats["qrels"] == 1
    assert stats["evidence_recoveries"] == 3
    assert stats["attribute_extractions"] == 11
    assert stats["query_rows_per_table"] == 5

    query = read_manifest_artifact(output_dir, "query_tables")[0]
    data_lake = read_manifest_artifact(output_dir, "data_lake_tables")
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    target = next(item for item in data_lake if item["role"] == "target_data_lake_table")
    rejected = next(item for item in data_lake if item["role"] == "raw_data_lake_table")
    recoveries = read_manifest_artifact(output_dir, "evidence_recoveries")
    extractions = read_manifest_artifact(output_dir, "attribute_extractions")
    qrels = [json.loads(line) for line in (output_dir / "qrels.jsonl").read_text(encoding="utf-8").splitlines()]

    query_column_names = [column["column_name"] for column in query["columns"]]
    target_column_names = [column["column_name"] for column in target["columns"]]
    assert query_column_names[0] == "Entity"
    assert set(query_column_names[1:]) | (set(target_column_names) - {"City"}) == {
        "Team",
        "League",
    }
    assert set(query_column_names).isdisjoint(target_column_names)
    assert query["hidden_attributes"][0]["column_name"] == "City"
    assert query["hidden_attributes"][0]["valid_entity_rows"] == 6
    assert query["hidden_attributes"][0]["required_recovered_rows"] == 3
    # Coverage is measured over the five rows exposed by this query view.
    assert query["hidden_attributes"][0]["recovered_value_ratio"] == 0.6
    assert query["hidden_attributes"][0]["selected_rows"] == 5
    assert query["hidden_attributes"][0]["target_rows"] == 6
    assert len(query["rows"]) == 5
    assert len(target["rows"]) == 6
    assert query["source_row_indices"] == [0, 1, 2, 3, 4]
    assert target["source_row_indices"] == [0, 1, 2, 3, 4, 5]
    city_position = target_column_names.index("City")
    assert [
        target["rows"][row_id]["cells"][city_position]["text"]
        for row_id in (1, 3)
    ] == ["", ""]
    assert rejected["queryable"] is False
    assert rejected["source_table_ref"] == {
        "artifact": "source_tables",
        "source_table_id": rejected["source_table_id"],
    }
    assert "rows" not in rejected
    resolved_data_lake = list(
        iter_manifest_records(
            output_dir,
            "data_lake_tables",
            log_every=0,
        )
    )
    resolved_rejected = next(
        item
        for item in resolved_data_lake
        if item["role"] == "raw_data_lake_table"
    )
    assert len(resolved_rejected["rows"]) == 5
    assert "source_table_ref" not in resolved_rejected
    assert manifest["query_construction"]["query_rows_per_table"] == 5
    assert (
        manifest["query_construction"]["query_row_selection"]
        == "recovery_balanced_disjoint_train_views"
    )
    assert manifest["query_construction"]["target_row_scope"] == "all_source_rows"
    assert manifest["query_construction"]["min_implicit_context_columns"] == 2
    assert qrels[0]["query_table_id"] == query["table_id"]
    assert qrels[0]["data_lake_table_id"] == target["table_id"]
    assert any(item["attributes"] for item in extractions)

    alpha = next(item for item in recoveries if item["query_entity"]["cell_text"] == "Alpha")
    assert alpha["source_row_id"] == 0
    assert alpha["query_row_id"] == 0
    assert alpha["recovered_attribute"] == {
        "column_index": 1,
        "column_name": "City",
        "value": "Paris",
        "model_value": "Paris",
        "hidden_in_query": True,
    }
    assert [node["node_type"] for node in alpha["path_nodes"]] == ["query_table", "text_asset", "target_table"]
    assert alpha["evidence"]["asset_type"] == "text"
    assert "City: Paris" in alpha["evidence"]["content_snippet"]
    assert alpha["evidence"]["model_evidence"] == ""
    assert alpha["evidence"]["model_connection_evidence"] == ""


def test_wikipedia_svg_download_converts_to_png_without_thumbnail(tmp_path, monkeypatch):
    svg_bytes = b'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800"></svg>'

    class FakeResponse:
        status_code = 200
        headers = {"Content-Type": "image/svg+xml"}
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_content(self, chunk_size):
            yield svg_bytes

    class FakeSession:
        def __init__(self):
            self.urls = []
            self.headers = {}

        def get(self, url, stream=True, timeout=60):
            self.urls.append(url)
            return FakeResponse()

    def fake_rasterize(svg_path, png_path, imageinfo):
        assert svg_path.read_bytes() == svg_bytes
        assert imageinfo["width"] == 1200
        png_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return True, "fake"

    monkeypatch.setattr(mm_table_dataset, "rasterize_svg_to_png", fake_rasterize)

    client = WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="test",
    )
    client.session = FakeSession()
    record = client.download_image(
        {
            "url": "https://example.test/original.svg",
            "mime": "image/svg+xml",
            "width": 1200,
            "height": 800,
        },
        "asset_svg",
    )

    assert record is not None
    assert record["file_name"].startswith("media_")
    assert record["file_name"].endswith(".png")
    assert Path(record["local_path"]).read_bytes().startswith(b"\x89PNG")
    assert record["converted_from"] == "image/svg+xml"
    assert record["source_bytes"] == len(svg_bytes)
    assert not list((tmp_path / "images").glob("*.tmp"))
    assert client.session.urls == ["https://example.test/original.svg"]


def test_wikipedia_svg_conversion_failure_removes_source_and_output_temps(
    tmp_path,
    monkeypatch,
):
    def failing_rasterize(svg_path, png_path, imageinfo):
        assert svg_path.exists()
        png_path.write_bytes(b"partial")
        return False, "bad svg"

    monkeypatch.setattr(
        mm_table_dataset,
        "rasterize_svg_to_png",
        failing_rasterize,
    )
    client = make_wikipedia_client(tmp_path)
    client.session = SequenceImageSession(
        [
            FakeStreamingResponse(
                200,
                b"<svg></svg>",
                {"Content-Type": "image/svg+xml"},
            )
        ]
    )

    record = client.download_image(
        {
            "file_title": "File:Broken.svg",
            "url": "https://example.test/Broken.svg",
            "mime": "image/svg+xml",
            "width": 1200,
            "height": 800,
        },
        "asset_broken_svg",
    )

    assert record is None
    assert not list((tmp_path / "images").glob("*.tmp"))
    assert not list((tmp_path / "images").glob("media_*.png"))


def test_svg_rasterization_handles_wikimedia_namespace_entities(tmp_path):
    pytest.importorskip("cairosvg")
    svg_path = tmp_path / "entity.svg"
    png_path = tmp_path / "entity.png"
    svg_path.write_text(
        """<!DOCTYPE svg [
<!ENTITY ns_svg "http://www.w3.org/2000/svg">
<!ENTITY ns_xlink "http://www.w3.org/1999/xlink">
]>
<svg xmlns="&ns_svg;" xmlns:xlink="&ns_xlink;" width="120" height="80">
  <rect width="120" height="80" fill="red"/>
</svg>
""",
        encoding="utf-8",
    )

    ok, reason = mm_table_dataset.rasterize_svg_to_png(
        svg_path,
        png_path,
        {"mime": "image/svg+xml", "width": 120, "height": 80},
    )

    assert ok
    assert reason == "cairosvg_sanitized_entities"
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_hitl_label_merge():
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        write_jsonl(stage / "hitl_pool.jsonl", [{"path_id": "p1", "human_label": None}])
        labels = stage / "labels.jsonl"
        write_jsonl(labels, [{"path_id": "p1", "label": 2, "annotator_notes": "ok"}])
        merge_human_labels(argparse.Namespace(stage1_dir=str(stage), human_labels=str(labels)))
        merged = [json.loads(line) for line in (stage / "human_labeled_paths.jsonl").read_text().splitlines()]
        assert merged[0]["human_label"] == 2
        assert merged[0]["label_source"] == "human"


def make_hitl_stage(stage: Path):
    input_dir = stage / "input"
    bridge_dir = input_dir / "bridge_assets"
    bridge_dir.mkdir(parents=True)
    write_jsonl(
        bridge_dir / "part-00000.jsonl",
        [
            {"asset_id": "a_new", "asset_type": "text", "content": "Alpha bridge evidence"},
            {"asset_id": "a_prev", "asset_type": "text", "content": "Previous evidence"},
            {"asset_id": "a_human", "asset_type": "text", "content": "Human evidence"},
        ],
    )
    (input_dir / "dataset_manifest.json").write_text(
        json.dumps({"artifacts": {"bridge_assets": {"shards": [{"path": "bridge_assets/part-00000.jsonl", "records": 3}]}}}),
        encoding="utf-8",
    )
    (stage / "manifest.json").write_text(json.dumps({"evidence_paths": {"input_dir": str(input_dir)}}), encoding="utf-8")
    fragment = {
        "fragment_id": "q",
        "columns": [{"column_name": "Entity"}],
        "rows": [{"cells": [{"text": "Alpha"}]}],
        "page_title": "Page",
    }
    target = {
        "fragment_id": "t",
        "columns": [{"column_name": "Bridge"}],
        "rows": [{"cells": [{"text": "Beta"}]}],
        "page_title": "Page",
    }
    write_jsonl(stage / "logic_fragments.jsonl", [fragment, target])
    pool = [
        {"path_id": "p_new", "query_fragment_id": "q", "target_fragment_id": "t", "asset_id": "a_new", "asset_type": "text", "split": "train", "bridge_col_name": "Bridge", "bridge_value": "Beta", "claim_text": "Alpha -> Beta", "weak_score": 0.5, "human_label": None},
        {"path_id": "p_prev", "query_fragment_id": "q", "target_fragment_id": "t", "asset_id": "a_prev", "asset_type": "text", "split": "train", "bridge_col_name": "Bridge", "bridge_value": "Beta", "claim_text": "Alpha -> Beta", "weak_score": 0.5, "human_label": None},
        {"path_id": "p_human", "query_fragment_id": "q", "target_fragment_id": "t", "asset_id": "a_human", "asset_type": "text", "split": "train", "bridge_col_name": "Bridge", "bridge_value": "Beta", "claim_text": "Alpha -> Beta", "weak_score": 0.5, "human_label": None},
    ]
    write_jsonl(stage / "hitl_pool.jsonl", pool)
    write_jsonl(stage / "hitl_selected_round_0.jsonl", [pool[1]])
    write_jsonl(stage / "human_labeled_paths.jsonl", [{**pool[2], "human_label": 2, "label_source": "human"}])


def test_hitl_selection_excludes_labeled_and_previous(tmp_path):
    make_hitl_stage(tmp_path)
    select_hitl_batch(
        argparse.Namespace(
            stage1_dir=str(tmp_path),
            teacher_scores=None,
            round_id=1,
            batch_size=10,
            candidate_top_n=10,
            seed=13,
            allow_reselect_previous=False,
            allow_reselect_labeled=False,
        )
    )
    selected = [json.loads(line) for line in (tmp_path / "hitl_selected_round_1.jsonl").read_text().splitlines()]
    assert [item["path_id"] for item in selected] == ["p_new"]


def test_hitl_template_preview_includes_bridge_entity_row(tmp_path):
    input_dir = tmp_path / "input"
    bridge_dir = input_dir / "bridge_assets"
    source_dir = input_dir / "source_tables"
    bridge_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    write_jsonl(bridge_dir / "part-00000.jsonl", [{"asset_id": "asset_focus", "asset_type": "text", "content": "Focus evidence"}])
    source_rows = [
        {
            "row_id": idx,
            "cells": [
                {"column_index": 0, "column_name": "Entity", "text": f"Entity {idx}", "wiki_title": f"Entity {idx}", "has_wiki_link": True},
                {"column_index": 1, "column_name": "Bridge", "text": "Focus Bridge" if idx == 0 else f"Bridge {idx}", "wiki_title": None, "has_wiki_link": False},
            ],
        }
        for idx in range(6)
    ] + [
        {
            "row_id": 6,
            "cells": [
                {"column_index": 0, "column_name": "Entity", "text": "Focus Entity", "wiki_title": "Focus Entity", "has_wiki_link": True},
                {"column_index": 1, "column_name": "Bridge", "text": "Focus Bridge", "wiki_title": None, "has_wiki_link": False},
            ],
        }
    ]
    write_jsonl(
        source_dir / "part-00000.jsonl",
        [
            {
                "source_table_id": "source_focus",
                "page_title": "Focus Page",
                "caption": "Focus caption",
                "columns": [{"column_index": 0, "column_name": "Entity"}, {"column_index": 1, "column_name": "Bridge"}],
                "rows": source_rows,
            }
        ],
    )
    (input_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "bridge_assets": {"shards": [{"path": "bridge_assets/part-00000.jsonl", "records": 1}]},
                    "source_tables": {"shards": [{"path": "source_tables/part-00000.jsonl", "records": 1}]},
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"evidence_paths": {"input_dir": str(input_dir)}}), encoding="utf-8")
    query_rows = [{"cells": [{"text": f"Entity {idx}", "wiki_title": f"Entity {idx}"}]} for idx in range(6)] + [{"cells": [{"text": "Focus Entity", "wiki_title": "Focus Entity"}]}]
    target_rows = [{"cells": [{"text": f"Bridge {idx}"}]} for idx in range(6)] + [{"cells": [{"text": "Focus Bridge"}]}]
    write_jsonl(
        tmp_path / "logic_fragments.jsonl",
        [
            {"fragment_id": "query", "source_table_id": "source_focus", "columns": [{"column_name": "Entity"}], "rows": query_rows, "page_title": "Focus Page"},
            {"fragment_id": "target", "source_table_id": "source_focus", "columns": [{"column_name": "Bridge"}], "rows": target_rows, "page_title": "Focus Page"},
        ],
    )
    write_jsonl(
        tmp_path / "hitl_pool.jsonl",
        [
            {
                "path_id": "path_focus",
                "source_table_id": "source_focus",
                "query_fragment_id": "query",
                "target_fragment_id": "target",
                "asset_id": "asset_focus",
                "asset_type": "text",
                "split": "train",
                "source_row_id": 6,
                "entity_text": "Focus Entity",
                "bridge_col_name": "Bridge",
                "bridge_value": "Focus Bridge",
                "claim_text": "Focus Entity -> Focus Bridge",
                "weak_score": 0.5,
                "human_label": None,
            }
        ],
    )

    select_hitl_batch(
        argparse.Namespace(
            stage1_dir=str(tmp_path),
            teacher_scores=None,
            round_id=0,
            batch_size=1,
            candidate_top_n=1,
            seed=13,
            allow_reselect_previous=False,
            allow_reselect_labeled=False,
        )
    )

    template = [json.loads(line) for line in (tmp_path / "human_labels_template_round_0.jsonl").read_text().splitlines()]
    item = template[0]
    assert len(item["query_fragment_preview"]["rows"]) == 5
    assert len(item["target_fragment_preview"]["rows"]) == 5
    assert any(row.get("Entity") == "Focus Entity" and row.get("_focus") for row in item["query_fragment_preview"]["rows"])
    assert any(row.get("Bridge") == "Focus Bridge" and row.get("_focus") for row in item["target_fragment_preview"]["rows"])
    assert set(item["source_table_preview"]) == {"columns", "rows"}
    assert len(item["source_table_preview"]["rows"]) == 7
    assert sum(1 for row in item["source_table_preview"]["rows"] if row.get("_focus")) == 1
    assert sum(1 for row in item["query_fragment_preview"]["rows"] if row.get("_focus")) == 1
    assert sum(1 for row in item["target_fragment_preview"]["rows"] if row.get("_focus")) == 1
    source_focus = item["source_table_preview"]["rows"][-1]
    assert source_focus["Entity"] == "Focus Entity"
    assert "_links" not in source_focus
    query_focus = next(row for row in item["query_fragment_preview"]["rows"] if row.get("Entity") == "Focus Entity")
    assert "_links" not in query_focus


def test_gui_host_resolution_and_lan_url(monkeypatch):
    assert resolve_gui_host(None, lan=False) == "127.0.0.1"
    assert resolve_gui_host(None, lan=True) == "0.0.0.0"
    assert resolve_gui_host("192.168.1.10", lan=True) == "192.168.1.10"
    monkeypatch.setattr("stage1_gui.guess_lan_ipv4", lambda: "192.168.1.20")
    message = format_gui_urls("Annotation GUI", "0.0.0.0", 7860)
    assert "Local browser: http://127.0.0.1:7860" in message
    assert "LAN devices:   http://192.168.1.20:7860" in message


def test_force_retrain_cleanup_preserves_prepared_data_and_human_labels(tmp_path):
    prepared = [
        "logic_fragments.jsonl",
        "logic_pairs.jsonl",
        "qrels.jsonl",
        "hitl_pool.jsonl",
        "weak_labeled_paths.jsonl",
    ]
    training_files = [
        "teacher_scores.jsonl",
        "train_pairs_round_0.jsonl",
        "student_train_pairs.jsonl",
        "hitl_selected_round_0.jsonl",
        "human_labels_template_round_0.jsonl",
        "hitl_round_0_annotation_status.json",
    ]
    for name in prepared + training_files + ["human_labeled_paths.jsonl"]:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "teacher_round_0").mkdir()
    (tmp_path / "teacher_round_0" / "teacher.pt").write_text("model", encoding="utf-8")
    (tmp_path / "student").mkdir()
    (tmp_path / "student" / "student.pt").write_text("model", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "logic_connectivity": {},
                "teacher": {},
                "student": {},
                "teacher_training_data": {},
                "hitl_round_0": {},
                "human_labels": {},
            }
        ),
        encoding="utf-8",
    )

    removed = clear_training_outputs(tmp_path, tmp_path / "student")

    assert removed
    for name in prepared + ["human_labeled_paths.jsonl"]:
        assert (tmp_path / name).exists()
    for name in training_files:
        assert not (tmp_path / name).exists()
    assert not (tmp_path / "teacher_round_0").exists()
    assert not (tmp_path / "student").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "logic_connectivity" in manifest
    assert "human_labels" in manifest
    assert "teacher" not in manifest
    assert "student" not in manifest
    assert "hitl_round_0" not in manifest


def test_force_retrain_cleanup_can_reset_human_labels(tmp_path):
    (tmp_path / "human_labeled_paths.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"human_labels": {}, "teacher": {}}), encoding="utf-8")

    clear_training_outputs(tmp_path, reset_human_labels=True)

    assert not (tmp_path / "human_labeled_paths.jsonl").exists()
    assert "human_labels" not in json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))


def test_hitl_annotation_app_saves_and_merges(tmp_path):
    make_hitl_stage(tmp_path)
    template = [
        {
            "path_id": "p_new",
            "query_fragment_preview": {"fragment_id": "q", "columns": ["Entity"], "rows": [{"Entity": "Alpha"}]},
            "target_fragment_preview": {"fragment_id": "t", "columns": ["Bridge"], "rows": [{"Bridge": "Beta"}]},
            "claim_text": "Alpha -> Beta",
            "asset_type": "text",
            "evidence_text_snippet": "Alpha bridge evidence",
            "image_local_path": "",
            "bridge_col_name": "Bridge",
            "bridge_value": "Beta",
            "label": "",
            "annotator_notes": "",
        }
    ]
    write_jsonl(tmp_path / "human_labels_template_round_1.jsonl", template)
    app = create_app(tmp_path, 1)
    client = app.test_client()
    resp = client.post("/api/label", json={"path_id": "p_new", "label": "2", "annotator_notes": "verified"})
    assert resp.status_code == 200
    resp = client.post("/api/merge")
    assert resp.status_code == 200
    merged = [json.loads(line) for line in (tmp_path / "human_labeled_paths.jsonl").read_text().splitlines()]
    by_id = {item["path_id"]: item for item in merged}
    assert by_id["p_new"]["human_label"] == 2
    assert by_id["p_new"]["annotator_notes"] == "verified"


def test_connection_viewer_groups_stage1_artifacts(tmp_path):
    fragment_base = {
        "object_type": "table_fragment",
        "split": "train",
        "chain_id": "c1",
        "source_table_id": "s1",
        "page_title": "Page",
        "columns": [{"column_name": "A"}],
        "rows": [{"cells": [{"text": "alpha"}]}],
    }
    write_jsonl(
        tmp_path / "logic_fragments.jsonl",
        [
            {**fragment_base, "fragment_id": "qv", "role": "left_visible", "statement": "A -> B"},
            {**fragment_base, "fragment_id": "qh", "role": "left_hidden", "statement": "A -> hidden(B)"},
            {**fragment_base, "fragment_id": "t", "role": "right_target", "statement": "B -> C"},
        ],
    )
    write_jsonl(tmp_path / "logic_pairs.jsonl", [{"pair_id": "p", "chain_id": "c1", "label": 1, "weight": 1.0, "reason": "visible_chain"}])
    write_jsonl(tmp_path / "qrels.jsonl", [{"chain_id": "c1", "query_id": "qv", "target_id": "t", "query_role": "left_visible", "target_role": "right_target", "rel": 3}])
    write_jsonl(tmp_path / "hitl_pool.jsonl", [{"path_id": "path", "chain_id": "c1", "asset_type": "text", "claim_text": "alpha -> beta", "entity_text": "alpha", "bridge_col_name": "B", "bridge_value": "beta"}])
    groups = load_groups(tmp_path, max_rows=5, max_evidence_paths=10)
    assert len(groups) == 1
    assert groups[0]["visible"]["fragment_id"] == "qv"
    assert groups[0]["evidence_paths"][0]["path_id"] == "path"
    client = create_connection_viewer_app(tmp_path, max_rows=5, max_evidence_paths=10).test_client()
    resp = client.get("/?q=alpha")
    assert resp.status_code == 200
    assert b"Stage-1 Connection Viewer" in resp.data


def test_connection_viewer_shows_only_referenced_text_chunk(tmp_path):
    input_dir = tmp_path / "input"
    bridge_dir = input_dir / "bridge_assets"
    bridge_dir.mkdir(parents=True)
    write_jsonl(
        bridge_dir / "part-00000.jsonl",
        [
            {
                "asset_id": "asset_text_alpha_000",
                "source_asset_id": "asset_text_alpha",
                "asset_type": "text",
                "entity_wiki_title": "Alpha",
                "content": "First chunk not used as the bridge.",
                "source": "wikipedia_extract_chunk",
                "text_chunk_index": 0,
                "text_chunk_count": 2,
            },
            {
                "asset_id": "asset_text_alpha_001",
                "source_asset_id": "asset_text_alpha",
                "asset_type": "text",
                "entity_wiki_title": "Alpha",
                "content": "Second chunk is the bridge evidence.",
                "source": "wikipedia_extract_chunk",
                "text_chunk_index": 1,
                "text_chunk_count": 2,
            },
        ],
    )
    (input_dir / "dataset_manifest.json").write_text(
        json.dumps({"artifacts": {"bridge_assets": {"shards": [{"path": "bridge_assets/part-00000.jsonl", "records": 2}]}}}),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"evidence_paths": {"input_dir": str(input_dir)}}), encoding="utf-8")
    fragment_base = {
        "object_type": "table_fragment",
        "split": "train",
        "chain_id": "c1",
        "source_table_id": "s1",
        "page_title": "Page",
        "columns": [{"column_name": "A"}],
        "rows": [{"cells": [{"text": "alpha"}]}],
    }
    write_jsonl(
        tmp_path / "logic_fragments.jsonl",
        [
            {**fragment_base, "fragment_id": "qv", "role": "left_visible"},
            {**fragment_base, "fragment_id": "qh", "role": "left_hidden"},
            {**fragment_base, "fragment_id": "t", "role": "right_target"},
        ],
    )
    write_jsonl(
        tmp_path / "evidence_paths.jsonl",
        [
            {
                "path_id": "path",
                "chain_id": "c1",
                "query_fragment_id": "qh",
                "target_fragment_id": "t",
                "asset_id": "asset_text_alpha_001",
                "asset_type": "text",
                "claim_text": "alpha -> beta",
            }
        ],
    )
    assets, loaded_input_dir = load_assets(tmp_path)
    groups = load_groups(
        tmp_path,
        max_rows=5,
        max_evidence_paths=10,
        assets=assets,
        input_dir=loaded_input_dir,
    )
    path = groups[0]["evidence_paths"][0]
    assert path["asset_content_snippet"] == "Second chunk is the bridge evidence."
    assert path["asset_chunk_label"] == "chunk 2/2"
    assert "First chunk" not in path["asset_content_snippet"]


def test_recall_viewer_cards_show_recalled_targets_and_bridge_path():
    fragments = {
        "q": {
            "fragment_id": "q",
            "role": "left_hidden",
            "split": "test",
            "chain_id": "c1",
            "page_title": "Query Page",
            "statement": "A -> hidden(B)",
            "columns": [{"column_name": "A"}],
            "rows": [{"cells": [{"text": "Alpha"}]}],
        },
        "t_hit": {
            "fragment_id": "t_hit",
            "role": "right_target",
            "split": "test",
            "chain_id": "c1",
            "page_title": "Target Page",
            "statement": "B -> C",
            "columns": [{"column_name": "B"}],
            "rows": [{"cells": [{"text": "Beta"}]}],
        },
        "t_miss": {
            "fragment_id": "t_miss",
            "role": "right_target",
            "split": "test",
            "chain_id": "c2",
            "page_title": "Other Page",
            "statement": "X -> Y",
            "columns": [{"column_name": "X"}],
            "rows": [{"cells": [{"text": "Other"}]}],
        },
    }
    qrels = [{"query_id": "q", "target_id": "t_hit", "rel": 2, "split": "test", "query_role": "left_hidden", "chain_id": "c1"}]
    direct_by_query = {"q": [{"target_id": "t_miss", "score": 0.8, "path": [("q", "table_fragment"), ("t_miss", "table_fragment")]}]}
    path_by_query = {
        "q": [
            {
                "target_id": "t_hit",
                "score": 0.7,
                "path": [("q", "table_fragment"), ("asset_a", "text_asset"), ("t_hit", "table_fragment")],
            }
        ]
    }
    path_records = {
        ("q", "asset_a", "t_hit"): {
            "path_id": "path_a",
            "query_fragment_id": "q",
            "asset_id": "asset_a",
            "target_fragment_id": "t_hit",
            "bridge_col_name": "Bridge",
            "bridge_value": "Beta",
        }
    }
    assets = {"asset_a": {"asset_id": "asset_a", "asset_type": "text", "entity_wiki_title": "Alpha", "content": "Alpha mentions Beta."}}

    cards = assemble_query_cards(qrels, fragments, direct_by_query, path_by_query, path_records, assets, max_rows=5)

    assert len(cards) == 1
    assert cards[0]["relevant_targets"][0]["target_fragment"]["rows"] == [{"B": "Beta"}]
    assert cards[0]["direct_targets"][0]["target_id"] == "t_miss"
    assert not cards[0]["direct_targets"][0]["is_relevant"]
    assert cards[0]["direct_targets"][0]["target_fragment"]["rows"] == [{"X": "Other"}]
    path_target = cards[0]["path_targets"][0]
    assert path_target["target_id"] == "t_hit"
    assert path_target["is_relevant"]
    assert path_target["target_fragment"]["rows"] == [{"B": "Beta"}]
    assert "<td>Beta</td>" in render_targets([path_target], "path")
    assert path_target["has_bridge"]
    bridge_node = path_target["path_nodes"][1]
    assert bridge_node["type"] == "text_asset"
    assert bridge_node["bridge"] == "Bridge = Beta"
    assert bridge_node["path_id"] == "path_a"


def test_eval_recall_records_can_drive_recall_viewer(tmp_path):
    write_jsonl(
        tmp_path / "logic_fragments.jsonl",
        [
            {
                "fragment_id": "q",
                "role": "left_hidden",
                "split": "test",
                "chain_id": "c1",
                "page_title": "Query Page",
                "statement": "A -> hidden(B)",
                "columns": [{"column_name": "A"}],
                "rows": [{"cells": [{"text": "Alpha"}]}],
            },
            {
                "fragment_id": "t",
                "role": "right_target",
                "split": "test",
                "chain_id": "c1",
                "page_title": "Target Page",
                "statement": "B -> C",
                "columns": [{"column_name": "B"}],
                "rows": [{"cells": [{"text": "Beta"}]}],
            },
        ],
    )
    qrels = [{"query_id": "q", "target_id": "t", "rel": 2, "split": "test", "query_role": "left_hidden", "chain_id": "c1"}]
    direct_records = direct_recall_records(qrels, {"q": ["t"]}, topk=1)
    bridge_record = bridge_recall_record(
        "q",
        qrels,
        ["t"],
        {"t": {"score": 0.75, "path": [("q", "table_fragment"), ("asset_a", "text_asset"), ("t", "table_fragment")]}},
        {
            ("q", "asset_a", "t"): {
                "path_id": "path_a",
                "asset_id": "asset_a",
                "asset_type": "text",
                "bridge_col_name": "Bridge",
                "bridge_value": "Beta",
                "claim_text": "Alpha -> Beta",
            }
        },
        {"asset_a": {"asset_id": "asset_a", "asset_type": "text", "title": "Alpha", "content_snippet": "Alpha mentions Beta."}},
        topk=1,
    )
    write_jsonl(tmp_path / "recall_rankings.jsonl", [*direct_records, bridge_record])
    args = argparse.Namespace(stage1_dir=str(tmp_path), recall_records=None, max_rows=5)

    cards = load_recorded_recall_cards(args)

    assert cards is not None
    assert cards[0]["direct_targets"][0]["target_id"] == "t"
    assert cards[0]["direct_targets"][0]["is_relevant"]
    path_target = cards[0]["path_targets"][0]
    assert path_target["score"] == 0.75
    assert path_target["path_nodes"][1]["bridge"] == "Bridge = Beta"
    assert path_target["path_nodes"][1]["path_id"] == "path_a"
    assert path_target["path_nodes"][1]["content"] == "Alpha mentions Beta."


def test_train_pairs_do_not_auto_positive_unlabeled_assets(tmp_path):
    stage = tmp_path
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {"fragment_id": "qv", "role": "left_visible", "object_type": "table_fragment", "split": "train", "chain_id": "c1", "source_table_id": "s1"},
            {"fragment_id": "qh", "role": "left_hidden", "object_type": "table_fragment", "split": "train", "chain_id": "c1", "source_table_id": "s1"},
            {"fragment_id": "t", "role": "right_target", "object_type": "table_fragment", "split": "train", "chain_id": "c1", "source_table_id": "s1"},
            {"fragment_id": "tneg", "role": "right_target", "object_type": "table_fragment", "split": "train", "chain_id": "c2", "source_table_id": "s2"},
        ],
    )
    write_jsonl(
        stage / "logic_pairs.jsonl",
        [{"pair_id": "p", "source_table_id": "s1", "split": "train", "chain_id": "c1", "query_fragment_id": "qv", "target_fragment_id": "t", "label": 1, "weight": 1.0}],
    )
    write_jsonl(
        stage / "hitl_pool.jsonl",
        [{"path_id": "path_unlabeled", "query_fragment_id": "qh", "asset_id": "a1", "asset_type": "text", "target_fragment_id": "t", "split": "train", "chain_id": "c1", "weak_label": None, "human_label": None}],
    )
    from build_teacher_training_data import run as build_train

    out = stage / "train_pairs.jsonl"
    build_train(
        argparse.Namespace(
            stage1_dir=str(stage),
            output=str(out),
            include_pseudo_labels="false",
            pseudo_pos_threshold=0.9,
            pseudo_neg_threshold=0.1,
            seed=13,
        )
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert not any(r.get("path_id") == "path_unlabeled" for r in records)


def test_table_only_train_pairs_exclude_path_labels(tmp_path):
    stage = tmp_path
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {"fragment_id": "qv", "role": "left_visible", "object_type": "table_fragment", "split": "train", "chain_id": "c1", "source_table_id": "s1"},
            {"fragment_id": "qh", "role": "left_hidden", "object_type": "table_fragment", "split": "train", "chain_id": "c1", "source_table_id": "s1"},
            {"fragment_id": "t", "role": "right_target", "object_type": "table_fragment", "split": "train", "chain_id": "c1", "source_table_id": "s1"},
            {"fragment_id": "tneg", "role": "right_target", "object_type": "table_fragment", "split": "train", "chain_id": "c2", "source_table_id": "s2"},
        ],
    )
    write_jsonl(
        stage / "logic_pairs.jsonl",
        [
            {"pair_id": "p_visible", "source_table_id": "s1", "split": "train", "chain_id": "c1", "query_fragment_id": "qv", "target_fragment_id": "t", "label": 1, "weight": 1.0},
            {"pair_id": "p_hidden", "source_table_id": "s1", "split": "train", "chain_id": "c1", "query_fragment_id": "qh", "target_fragment_id": "t", "label": 1, "weight": 0.4},
        ],
    )
    write_jsonl(
        stage / "hitl_pool.jsonl",
        [{"path_id": "path_weak", "query_fragment_id": "qv", "asset_id": "a1", "asset_type": "text", "target_fragment_id": "t", "split": "train", "chain_id": "c1", "weak_label": "weak_direct"}],
    )
    write_jsonl(
        stage / "human_labeled_paths.jsonl",
        [{"path_id": "path_human", "query_fragment_id": "qv", "asset_id": "a2", "asset_type": "image", "target_fragment_id": "t", "split": "train", "chain_id": "c1", "human_label": 2}],
    )
    write_jsonl(
        stage / "teacher_scores.jsonl",
        [{"sample_kind": "path_score", "path_id": "path_pseudo", "query_fragment_id": "qv", "asset_id": "a3", "asset_object_type": "text_asset", "target_fragment_id": "t", "path_score": 0.99, "split": "train", "chain_id": "c1"}],
    )
    from build_teacher_training_data import run as build_train

    out = stage / "train_pairs.jsonl"
    build_train(
        argparse.Namespace(
            stage1_dir=str(stage),
            output=str(out),
            include_pseudo_labels="true",
            pseudo_pos_threshold=0.9,
            pseudo_neg_threshold=0.1,
            seed=13,
            table_only=True,
        )
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records
    assert {r["sample_kind"] for r in records} == {"pair"}
    assert {r["object_id_a"] for r in records} == {"qv"}
    assert not any("asset_id" in r for r in records)


def test_table_only_train_pairs_include_same_source_non_joinable_hard_negatives(tmp_path):
    stage = tmp_path
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {
                "fragment_id": "qv",
                "role": "left_visible",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c1",
                "source_table_id": "s1",
                "visible_bridge_col": 1,
                "source_column_indices": [0, 1],
            },
            {
                "fragment_id": "t_pos",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c1",
                "source_table_id": "s1",
                "source_column_indices": [1, 2],
            },
            {
                "fragment_id": "t_hard",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c2",
                "source_table_id": "s1",
                "source_column_indices": [3],
            },
            {
                "fragment_id": "t_joinable",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c3",
                "source_table_id": "s1",
                "source_column_indices": [1, 4],
            },
            {
                "fragment_id": "t_other_source",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c4",
                "source_table_id": "s2",
                "source_column_indices": [9],
            },
        ],
    )
    write_jsonl(
        stage / "logic_pairs.jsonl",
        [
            {
                "pair_id": "p_visible",
                "source_table_id": "s1",
                "split": "train",
                "chain_id": "c1",
                "query_fragment_id": "qv",
                "target_fragment_id": "t_pos",
                "label": 1,
                "weight": 1.0,
            },
        ],
    )
    from build_teacher_training_data import run as build_train

    out = stage / "train_pairs.jsonl"
    build_train(
        argparse.Namespace(
            stage1_dir=str(stage),
            output=str(out),
            include_pseudo_labels="false",
            pseudo_pos_threshold=0.9,
            pseudo_neg_threshold=0.1,
            seed=13,
            table_only=True,
            hard_negatives_per_positive=3,
        )
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    hard = [record for record in records if record.get("label_source") == "hard_negative"]
    assert len(hard) == 1
    assert hard[0]["object_id_b"] == "t_hard"
    assert hard[0]["label"] == 0.0
    assert hard[0]["reason"] == "same_source_non_joinable_target"
    assert not any(record["object_id_b"] == "t_joinable" and record.get("label_source") == "hard_negative" for record in records)


def test_hidden_train_pairs_include_same_source_non_joinable_hard_negatives(tmp_path):
    stage = tmp_path
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {
                "fragment_id": "qh",
                "role": "left_hidden",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c1",
                "source_table_id": "s1",
                "hidden_bridge_col": 1,
                "source_column_indices": [0],
            },
            {
                "fragment_id": "t_pos",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c1",
                "source_table_id": "s1",
                "source_column_indices": [1, 2],
            },
            {
                "fragment_id": "t_hard",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c2",
                "source_table_id": "s1",
                "source_column_indices": [3],
            },
            {
                "fragment_id": "t_joinable",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c3",
                "source_table_id": "s1",
                "source_column_indices": [1, 4],
            },
            {
                "fragment_id": "t_other_source",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c4",
                "source_table_id": "s2",
                "source_column_indices": [9],
            },
        ],
    )
    write_jsonl(
        stage / "logic_pairs.jsonl",
        [
            {
                "pair_id": "p_hidden",
                "source_table_id": "s1",
                "split": "train",
                "chain_id": "c1",
                "query_fragment_id": "qh",
                "target_fragment_id": "t_pos",
                "label": 1,
                "weight": 0.4,
            },
        ],
    )
    from build_teacher_training_data import run as build_train

    out = stage / "train_pairs.jsonl"
    build_train(
        argparse.Namespace(
            stage1_dir=str(stage),
            output=str(out),
            include_pseudo_labels="false",
            pseudo_pos_threshold=0.9,
            pseudo_neg_threshold=0.1,
            seed=13,
            hard_negatives_per_positive=3,
        )
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    hard = [record for record in records if record.get("label_source") == "hard_negative"]
    assert len(hard) == 1
    assert hard[0]["object_id_a"] == "qh"
    assert hard[0]["object_id_b"] == "t_hard"
    assert not any(record["object_id_b"] == "t_joinable" and record.get("label_source") == "hard_negative" for record in records)


def test_hard_negatives_per_positive_zero_disables_hard_negatives(tmp_path):
    stage = tmp_path
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {
                "fragment_id": "qv",
                "role": "left_visible",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c1",
                "source_table_id": "s1",
                "visible_bridge_col": 1,
                "source_column_indices": [0, 1],
            },
            {
                "fragment_id": "t_pos",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c1",
                "source_table_id": "s1",
                "source_column_indices": [1, 2],
            },
            {
                "fragment_id": "t_hard",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c2",
                "source_table_id": "s1",
                "source_column_indices": [3],
            },
            {
                "fragment_id": "t_other_source",
                "role": "right_target",
                "object_type": "table_fragment",
                "split": "train",
                "chain_id": "c3",
                "source_table_id": "s2",
                "source_column_indices": [9],
            },
        ],
    )
    write_jsonl(
        stage / "logic_pairs.jsonl",
        [
            {
                "pair_id": "p_visible",
                "source_table_id": "s1",
                "split": "train",
                "chain_id": "c1",
                "query_fragment_id": "qv",
                "target_fragment_id": "t_pos",
                "label": 1,
                "weight": 1.0,
            },
        ],
    )
    from build_teacher_training_data import run as build_train

    out = stage / "train_pairs.jsonl"
    build_train(
        argparse.Namespace(
            stage1_dir=str(stage),
            output=str(out),
            include_pseudo_labels="false",
            pseudo_pos_threshold=0.9,
            pseudo_neg_threshold=0.1,
            seed=13,
            hard_negatives_per_positive=0,
        )
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert not any(record.get("label_source") == "hard_negative" for record in records)
    assert any(record.get("label_source") == "negative" for record in records)


def test_table_only_student_distill_records_exclude_paths(tmp_path):
    stage = tmp_path
    scores = stage / "teacher_scores.jsonl"
    write_jsonl(
        scores,
        [
            {"sample_kind": "pair_score", "query_id": "q", "target_id": "t", "score": 0.8},
            {"sample_kind": "path_score", "query_fragment_id": "q", "asset_id": "a", "asset_object_type": "text_asset", "target_fragment_id": "t", "score_Q_asset": 0.7, "score_asset_T": 0.6, "path_score": 0.6},
        ],
    )
    write_jsonl(
        stage / "human_labeled_paths.jsonl",
        [{"query_fragment_id": "q", "asset_id": "a2", "asset_type": "image", "target_fragment_id": "t", "human_label": 2}],
    )
    records = build_distill_records(stage, scores, table_only=True)
    assert records == [
        {"kind": "pair", "a": "q", "ta": "table_fragment", "b": "t", "tb": "table_fragment", "target": 0.8, "group_id": "pair:q:table_fragment"}
    ]


def test_table_only_teacher_scores_training_pair_negatives(tmp_path):
    write_jsonl(tmp_path / "qrels.jsonl", [{"query_id": "q", "target_id": "heldout"}])
    model = TeacherMLP(2)
    vectors = {
        "q": np.array([1.0, 0.0], dtype="float32"),
        "t_pos": np.array([1.0, 0.0], dtype="float32"),
        "t_neg": np.array([0.0, 1.0], dtype="float32"),
    }
    pair_records = [
        {"sample_kind": "pair", "object_id_a": "q", "object_type_a": "table_fragment", "object_id_b": "t_pos", "object_type_b": "table_fragment", "label": 1.0, "split": "train"},
        {"sample_kind": "pair", "object_id_a": "q", "object_type_a": "table_fragment", "object_id_b": "t_neg", "object_type_b": "table_fragment", "label": 0.0, "split": "train"},
    ]
    scores = score_paths(tmp_path, model, vectors, torch.device("cpu"), "min", table_only=True, pair_records=pair_records)
    assert {score["target_id"] for score in scores} == {"t_pos", "t_neg"}
    assert all(score["sample_kind"] == "pair_score" for score in scores)
    assert "heldout" not in {score["target_id"] for score in scores}


def test_student_ranking_groups_and_pairwise_loss():
    vectors = {
        "q": np.array([1.0, 0.0], dtype="float32"),
        "a": np.array([1.0, 0.0], dtype="float32"),
        "b": np.array([0.0, 1.0], dtype="float32"),
    }
    records = [
        {"kind": "pair", "a": "q", "ta": "table_fragment", "b": "a", "tb": "text_asset", "target": 0.9, "group_id": "g"},
        {"kind": "pair", "a": "q", "ta": "table_fragment", "b": "b", "tb": "text_asset", "target": 0.1, "group_id": "g"},
    ]
    groups = build_ranking_groups(records, vectors)
    assert len(groups) == 1
    model = Student(2, 2)
    args = argparse.Namespace(distill_loss="pairwise", max_pairs_per_group=100, pairwise_min_delta=1e-4, ranking_temperature=1.0)
    loss = train_loss(model, groups, vectors, torch.device("cpu"), args)
    assert float(loss.detach()) > 0.0


def test_relation_query_uses_relation_matrix():
    model = Student(2, 2)
    with torch.no_grad():
        for proj in model.proj.values():
            proj.weight.copy_(torch.eye(2))
        model.rel["table_fragment__text_asset"].copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    query = relation_query_from_projected(
        model,
        np.array([1.0, 0.0], dtype="float32"),
        "table_fragment",
        "text_asset",
        torch.device("cpu"),
    )[0]
    assert np.allclose(query, np.array([0.0, 1.0], dtype="float32"), atol=1e-6)


def test_table_only_eval_uses_hnsw_table_index(tmp_path):
    hnswlib = pytest.importorskip("hnswlib")
    stage = tmp_path
    student_dir = stage / "student"
    index_emb_dir = student_dir / "index_embeddings"
    hnsw_dir = stage / "hnsw_indices"
    index_emb_dir.mkdir(parents=True)
    hnsw_dir.mkdir()
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {"fragment_id": "qv", "role": "left_visible", "object_type": "table_fragment"},
            {"fragment_id": "qh", "role": "left_hidden", "object_type": "table_fragment"},
            {"fragment_id": "t_good", "role": "right_target", "object_type": "table_fragment"},
            {"fragment_id": "t_bad", "role": "right_target", "object_type": "table_fragment"},
        ],
    )
    projected_ids = ["qv", "qh", "t_good", "t_bad"]
    projected_arr = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype="float32")
    np.save(index_emb_dir / "table_fragment.npy", projected_arr)
    (index_emb_dir / "table_fragment_ids.json").write_text(json.dumps(projected_ids), encoding="utf-8")
    index_ids = ["t_good", "t_bad"]
    index_arr = projected_arr[[2, 3]]
    index = hnswlib.Index(space="cosine", dim=2)
    index.init_index(max_elements=2, ef_construction=20, M=8)
    index.add_items(index_arr, np.arange(2))
    index.save_index(str(hnsw_dir / "table_fragment.bin"))
    (hnsw_dir / "table_fragment_ids.json").write_text(json.dumps(index_ids), encoding="utf-8")
    model = Student(2, 2)
    with torch.no_grad():
        for proj in model.proj.values():
            proj.weight.copy_(torch.eye(2))
        model.rel["table_fragment__table_fragment"].copy_(torch.eye(2))
    rankings = table_rankings(
        argparse.Namespace(
            student_dir=str(student_dir),
            hnsw_dir=str(hnsw_dir),
            table_only=True,
            table_hnsw_k=0,
            progress=False,
            data_lake_split_mode="query_corpus",
        ),
        stage,
        model,
        {oid: projected_arr[idx] for idx, oid in enumerate(projected_ids)},
        torch.device("cpu"),
        [1],
    )
    assert rankings["qv"][0] == "t_good"
    assert "qh" not in rankings
    assert "qv" not in rankings["qv"]


def test_eval_infers_table_only_from_hnsw_stats(tmp_path):
    hnsw_dir = tmp_path / "hnsw_indices"
    hnsw_dir.mkdir()
    (hnsw_dir / "hnsw_stats.json").write_text(json.dumps({"table_only": True}), encoding="utf-8")
    args = argparse.Namespace(table_only=False, hnsw_dir=str(hnsw_dir))
    assert infer_table_only(args, tmp_path)


def test_eval_infers_raw_embedding_hnsw_from_stats(tmp_path):
    hnsw_dir = tmp_path / "hnsw_indices"
    hnsw_dir.mkdir()
    (hnsw_dir / "hnsw_stats.json").write_text(json.dumps({"embedding_backend": "raw"}), encoding="utf-8")
    args = argparse.Namespace(raw_embedding_hnsw=False, hnsw_dir=str(hnsw_dir))
    assert infer_raw_embedding_hnsw(args, tmp_path)


def test_table_only_direct_eval_qrels_keep_only_visible_queries():
    qrels = [
        {"query_id": "qv", "target_id": "t", "query_role": "left_visible"},
        {"query_id": "qh", "target_id": "t", "query_role": "left_hidden"},
    ]
    assert direct_eval_qrels(qrels, table_only=True) == [qrels[0]]
    assert direct_eval_qrels(qrels, table_only=False) == qrels


def test_path_aware_recall_fraction_counts_all_relevant_targets():
    ranked = ["t1", "bad", "t2"]
    correct = {"t1", "t2", "t3"}
    assert recall_fraction(ranked, correct, 1) == pytest.approx(1 / 3)
    assert recall_fraction(ranked, correct, 3) == pytest.approx(2 / 3)


def test_hnsw_table_index_contains_only_right_targets(tmp_path):
    pytest.importorskip("hnswlib")
    from build_hnsw_indices import run as build_hnsw

    stage = tmp_path
    student_dir = stage / "student"
    emb_dir = student_dir / "index_embeddings"
    emb_dir.mkdir(parents=True)
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {"fragment_id": "qv", "role": "left_visible", "object_type": "table_fragment"},
            {"fragment_id": "qh", "role": "left_hidden", "object_type": "table_fragment"},
            {"fragment_id": "t", "role": "right_target", "object_type": "table_fragment"},
        ],
    )
    np.save(emb_dir / "table_fragment.npy", np.eye(3, dtype="float32"))
    (emb_dir / "table_fragment_ids.json").write_text(json.dumps(["qv", "qh", "t"]), encoding="utf-8")
    build_hnsw(
        argparse.Namespace(
            stage1_dir=str(stage),
            student_dir=str(student_dir),
            space="cosine",
            m=8,
            ef_construction=20,
            ef_search=20,
            table_only=True,
            data_lake_split_mode="query_corpus",
        )
    )
    indexed_ids = json.loads((stage / "hnsw_indices" / "table_fragment_ids.json").read_text(encoding="utf-8"))
    stats = json.loads((stage / "hnsw_indices" / "hnsw_stats.json").read_text(encoding="utf-8"))
    assert indexed_ids == ["t"]
    assert stats["objects"][0]["indexed_role"] == "right_target"


def test_raw_embedding_hnsw_index_uses_embedding_dir(tmp_path):
    pytest.importorskip("hnswlib")
    from build_hnsw_indices import run as build_hnsw

    stage = tmp_path
    emb_dir = stage / "embeddings"
    emb_dir.mkdir()
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {"fragment_id": "qv", "role": "left_visible", "object_type": "table_fragment"},
            {"fragment_id": "t", "role": "right_target", "object_type": "table_fragment"},
        ],
    )
    np.save(emb_dir / "table_fragment.npy", np.eye(2, dtype="float32"))
    (emb_dir / "table_fragment_ids.json").write_text(json.dumps(["qv", "t"]), encoding="utf-8")
    build_hnsw(
        argparse.Namespace(
            stage1_dir=str(stage),
            student_dir=str(stage / "student"),
            embedding_dir=str(emb_dir),
            hnsw_dir=str(stage / "hnsw_indices"),
            space="cosine",
            m=8,
            ef_construction=20,
            ef_search=20,
            table_only=True,
            raw_embedding_hnsw=True,
            data_lake_split_mode="query_corpus",
        )
    )
    indexed_ids = json.loads((stage / "hnsw_indices" / "table_fragment_ids.json").read_text(encoding="utf-8"))
    stats = json.loads((stage / "hnsw_indices" / "hnsw_stats.json").read_text(encoding="utf-8"))
    assert indexed_ids == ["t"]
    assert stats["embedding_backend"] == "raw"
    assert stats["embedding_dir"] == str(emb_dir)


def test_strict_hnsw_table_index_is_split_specific(tmp_path):
    pytest.importorskip("hnswlib")
    from build_hnsw_indices import run as build_hnsw

    stage = tmp_path
    emb_dir = stage / "embeddings"
    emb_dir.mkdir()
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {"fragment_id": "q_train", "role": "left_visible", "object_type": "table_fragment", "split": "train"},
            {"fragment_id": "q_test", "role": "left_visible", "object_type": "table_fragment", "split": "test"},
            {"fragment_id": "t_train", "role": "right_target", "object_type": "table_fragment", "split": "train"},
            {"fragment_id": "t_test", "role": "right_target", "object_type": "table_fragment", "split": "test"},
        ],
    )
    ids = ["q_train", "q_test", "t_train", "t_test"]
    np.save(emb_dir / "table_fragment.npy", np.eye(4, dtype="float32"))
    (emb_dir / "table_fragment_ids.json").write_text(json.dumps(ids), encoding="utf-8")

    build_hnsw(
        argparse.Namespace(
            stage1_dir=str(stage),
            student_dir=str(stage / "student"),
            embedding_dir=str(emb_dir),
            hnsw_dir=str(stage / "hnsw_indices"),
            space="cosine",
            m=8,
            ef_construction=20,
            ef_search=20,
            table_only=True,
            raw_embedding_hnsw=True,
            data_lake_split_mode="strict",
        )
    )

    hnsw_dir = stage / "hnsw_indices"
    stats = json.loads((hnsw_dir / "hnsw_stats.json").read_text(encoding="utf-8"))

    assert not (hnsw_dir / "table_fragment.bin").exists()
    assert json.loads((hnsw_dir / "table_fragment_train_ids.json").read_text(encoding="utf-8")) == ["t_train"]
    assert json.loads((hnsw_dir / "table_fragment_test_ids.json").read_text(encoding="utf-8")) == ["t_test"]
    assert {(item["split"], item["index_name"]) for item in stats["objects"]} == {
        ("train", "table_fragment_train"),
        ("test", "table_fragment_test"),
    }


def test_strict_raw_embedding_table_rankings_use_query_split_index(tmp_path):
    pytest.importorskip("hnswlib")
    from build_hnsw_indices import run as build_hnsw

    stage = tmp_path
    emb_dir = stage / "embeddings"
    emb_dir.mkdir()
    fragments = [
        {"fragment_id": "q_train", "role": "left_visible", "object_type": "table_fragment", "split": "train"},
        {"fragment_id": "q_test", "role": "left_visible", "object_type": "table_fragment", "split": "test"},
        {"fragment_id": "t_train", "role": "right_target", "object_type": "table_fragment", "split": "train"},
        {"fragment_id": "t_test", "role": "right_target", "object_type": "table_fragment", "split": "test"},
    ]
    write_jsonl(stage / "logic_fragments.jsonl", fragments)
    ids = ["q_train", "q_test", "t_train", "t_test"]
    arr = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype="float32",
    )
    np.save(emb_dir / "table_fragment.npy", arr)
    (emb_dir / "table_fragment_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    build_hnsw(
        argparse.Namespace(
            stage1_dir=str(stage),
            student_dir=str(stage / "student"),
            embedding_dir=str(emb_dir),
            hnsw_dir=str(stage / "hnsw_indices"),
            space="cosine",
            m=8,
            ef_construction=20,
            ef_search=20,
            table_only=True,
            raw_embedding_hnsw=True,
            data_lake_split_mode="strict",
        )
    )

    rankings = table_rankings(
        argparse.Namespace(
            embedding_dir=str(emb_dir),
            student_dir=str(stage / "student"),
            hnsw_dir=str(stage / "hnsw_indices"),
            table_only=True,
            raw_embedding_hnsw=True,
            table_hnsw_k=1,
            progress=False,
            data_lake_split_mode="strict",
        ),
        stage,
        None,
        {oid: arr[idx] for idx, oid in enumerate(ids)},
        torch.device("cpu"),
        [1],
    )

    assert rankings["q_test"] == ["t_test"]
    assert rankings["q_train"] == ["t_train"]


def test_raw_embedding_table_rankings_use_hnsw_without_student(tmp_path):
    hnswlib = pytest.importorskip("hnswlib")
    stage = tmp_path
    emb_dir = stage / "embeddings"
    hnsw_dir = stage / "hnsw_indices"
    emb_dir.mkdir()
    hnsw_dir.mkdir()
    write_jsonl(
        stage / "logic_fragments.jsonl",
        [
            {"fragment_id": "qv", "role": "left_visible", "object_type": "table_fragment"},
            {"fragment_id": "qh", "role": "left_hidden", "object_type": "table_fragment"},
            {"fragment_id": "t_good", "role": "right_target", "object_type": "table_fragment"},
            {"fragment_id": "t_bad", "role": "right_target", "object_type": "table_fragment"},
        ],
    )
    ids = ["qv", "qh", "t_good", "t_bad"]
    arr = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype="float32")
    np.save(emb_dir / "table_fragment.npy", arr)
    (emb_dir / "table_fragment_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    index = hnswlib.Index(space="cosine", dim=2)
    index.init_index(max_elements=2, ef_construction=20, M=8)
    index.add_items(arr[[2, 3]], np.arange(2))
    index.save_index(str(hnsw_dir / "table_fragment.bin"))
    (hnsw_dir / "table_fragment_ids.json").write_text(json.dumps(["t_good", "t_bad"]), encoding="utf-8")
    rankings = table_rankings(
        argparse.Namespace(
            embedding_dir=str(emb_dir),
            student_dir=str(stage / "student"),
            hnsw_dir=str(hnsw_dir),
            table_only=False,
            raw_embedding_hnsw=True,
            table_hnsw_k=0,
            progress=False,
            data_lake_split_mode="query_corpus",
        ),
        stage,
        None,
        {oid: arr[idx] for idx, oid in enumerate(ids)},
        torch.device("cpu"),
        [1],
    )
    assert rankings["qv"][0] == "t_good"
    assert rankings["qh"][0] == "t_good"
    assert "qv" not in rankings["qv"]


def test_raw_embedding_beam_search_keeps_multihop_recall(tmp_path):
    hnswlib = pytest.importorskip("hnswlib")
    hnsw_dir = tmp_path / "hnsw_indices"
    hnsw_dir.mkdir()
    vectors = {
        "qh": np.array([1.0, 0.0], dtype="float32"),
        "txt": np.array([0.8, 0.6], dtype="float32"),
        "t": np.array([0.7, 0.7], dtype="float32"),
    }
    ids_by_type = {
        "table_fragment": ["t"],
        "text_asset": ["txt"],
    }
    indexes = {}
    for object_type, ids in ids_by_type.items():
        arr = np.stack([vectors[oid] for oid in ids]).astype("float32")
        index = hnswlib.Index(space="cosine", dim=2)
        index.init_index(max_elements=len(ids), ef_construction=20, M=8)
        index.add_items(arr, np.arange(len(ids)))
        index.save_index(str(hnsw_dir / f"{object_type}.bin"))
        indexes[object_type] = index
    projected = {
        "qh": ("table_fragment", vectors["qh"]),
        "txt": ("text_asset", vectors["txt"]),
        "t": ("table_fragment", vectors["t"]),
    }
    ranked, best_paths = beam_search_tables(
        argparse.Namespace(
            raw_embedding_hnsw=True,
            beam_neighbors=1,
            max_hops=2,
            beam_width=4,
            path_composition="min",
        ),
        None,
        vectors,
        projected,
        indexes,
        ids_by_type,
        "qh",
        torch.device("cpu"),
    )
    assert ranked[0] == "t"
    assert best_paths["t"]["path"] == [("qh", "table_fragment"), ("txt", "text_asset"), ("t", "table_fragment")]


def test_hnsw_index_build_query():
    hnswlib = pytest.importorskip("hnswlib")
    arr = np.eye(4, dtype="float32")
    index = hnswlib.Index(space="cosine", dim=4)
    index.init_index(max_elements=4, ef_construction=20, M=8)
    index.add_items(arr, np.arange(4))
    labels, _ = index.knn_query(arr[0], k=1)
    assert int(labels[0][0]) == 0

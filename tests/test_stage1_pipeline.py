import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_mm_table_dataset as mm_table_dataset
import build_stage1_embeddings as stage1_embeddings
from build_stage1_logic_connectivity import build_target_rows, choose_query_context_cols
from build_mm_table_dataset import ShardedJsonlWriter, WikipediaClient, build_bridge_assets, split_text_asset_content
from hitl_annotation_app import create_app
from stage1_connection_viewer import create_app as create_connection_viewer_app
from stage1_connection_viewer import load_assets
from stage1_connection_viewer import load_groups
from merge_human_labels import run as merge_human_labels
from select_hitl_batch import run as select_hitl_batch
from qwen3_vl_embedding import Qwen3VLEmbeddingEncoder, image_limit_from_exception, resize_batch_images_for_limit
from stage1_io import fd_purity, project_rows, write_jsonl
from stage1_serialization import serialize_table_for_embedding
from train_student import Student, build_ranking_groups, train_loss
from eval_stage1_recall import relation_query_from_projected


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
    assert "Argentina" not in text
    assert "Country" not in text
    assert "Player" in text


def test_qwen_encoder_wrapper_dummy_text_mock_normalized():
    encoder = Qwen3VLEmbeddingEncoder(mock=True)
    arr = encoder.encode_texts(["hello", "world"])
    assert arr.shape == (2, 32)
    assert np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-5)


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


def test_wikipedia_svg_download_converts_to_png_without_thumbnail(tmp_path, monkeypatch):
    svg_bytes = b'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800"></svg>'

    class FakeResponse:
        headers = {"Content-Type": "image/svg+xml"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

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
    assert record["file_name"] == "asset_svg.png"
    assert Path(record["local_path"]).read_bytes().startswith(b"\x89PNG")
    assert record["converted_from"] == "image/svg+xml"
    assert record["source_bytes"] == len(svg_bytes)
    assert not (tmp_path / "images" / "asset_svg.svg.tmp").exists()
    assert client.session.urls == ["https://example.test/original.svg"]


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


def test_hnsw_index_build_query():
    hnswlib = pytest.importorskip("hnswlib")
    arr = np.eye(4, dtype="float32")
    index = hnswlib.Index(space="cosine", dim=4)
    index.init_index(max_elements=4, ef_construction=20, M=8)
    index.add_items(arr, np.arange(4))
    labels, _ = index.knn_query(arr[0], k=1)
    assert int(labels[0][0]) == 0

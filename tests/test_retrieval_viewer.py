import gzip
import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrieval_viewer import assemble_query, compact_table, make_handler, write_json


def test_gold_path_is_pair_specific_and_d1_does_not_imply_c150():
    pool = {"query_id": "q", "D100_ANN": ["wrong"], "C150": ["wrong"], "U": ["wrong", "gold"],
            "QT_ranks": {"wrong": 1, "gold": 2}, "D1_ranks": {"gold": 1},
            "all_U_QT_scores": {"wrong": .8, "gold": .7}, "retained_bags": {"gold": ["e"], "wrong": ["e"]}}
    first = [{"evidence_id": "e", "modality": "image", "rank": 2, "score": .9}]
    paths = [{"query_id": "q", "evidence_id": "e", "target_id": target, "first_rank": 2,
              "second_rank": rank, "second_raw_score": score, "path_raw_score": .9 + score,
              "retained": True} for target, rank, score in [("gold", 7, .4), ("wrong", 1, .8)]]
    result = assemble_query(pool, first, paths, [{"target_id": "gold", "evidence_ids": ["e"]}], {})
    targets = result["evidence"][0]["targets"]
    assert [(r["id"], r["rank"], r["gold_path"]) for r in targets] == [("wrong", 1, False), ("gold", 7, True)]
    assert result["evidence"][0]["rank"] == 2
    assert targets[1]["retained"] and result["targets"]["gold"]["c150_rank"] is None
    assert not result["teacher_available"]


def test_compact_table_preserves_local_order_and_only_visible_cells():
    table = {"table_id": "q", "source_table_id": "source", "columns": [
        {"column_index": 2, "column_name": "title"}, {"column_index": 0, "column_name": "price"}],
        "rows": [{"row_id": 42, "cells": [{"column_index": 0, "text": "12.3"}, {"column_index": 2, "text": "Book"}]}],
        "hidden_attributes": [{"column_name": "authors", "value": "hidden"}]}
    result = compact_table(table)
    assert result["rows"] == [["Book", "12.3"]]
    assert result["row_ids"] == [42]
    assert result["hidden"] == ["authors"]


def test_http_serves_exported_data_and_registered_images_only(tmp_path):
    exported = tmp_path / "viewer"
    image = tmp_path / "image.png"
    image.write_bytes(b"fake-image")
    write_json(exported / "image_paths.json", {"ev:known": str(image)})
    write_json(exported / "data/catalog.json.gz", {"queries": 150})
    write_json(exported / "outside.json.gz", {"must_not_be_served": True})
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(exported))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = build_opener(ProxyHandler({}))
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with opener.open(url + "/data/catalog.json.gz") as response:
            assert response.headers["Content-Encoding"] == "gzip"
            assert json.loads(gzip.decompress(response.read())) == {"queries": 150}
        with opener.open(url + "/image/ev%3Aknown") as response:
            assert response.read() == b"fake-image"
        for path in ["/image_paths.json", "/data/%2e%2e/outside.json.gz", "/image/unknown", "/src/", "/"]:
            if path == "/":
                with opener.open(url + path) as response:
                    assert b"<!doctype html>" in response.read()
            else:
                with pytest.raises(HTTPError) as error:
                    opener.open(url + path)
                assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

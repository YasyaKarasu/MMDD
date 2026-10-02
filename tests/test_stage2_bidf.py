"""Behaviour of the Stage-2 B+IDF pipeline pieces on small synthetic inputs (CPU, no models)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mmdd_stage2 import localizer, matching, recovery, selector, stage1, values, visible
from mmdd_stage2.evaluate import metrics


def rows(*columns: tuple[str, list[str]]) -> list[dict]:
    """Five query rows from ``(column name, five texts)`` pairs."""
    return [{"query_row_id": r, "cells": [{"column_id": i, "column_name": name, "text": texts[r]}
                                           for i, (name, texts) in enumerate(columns)]} for r in range(5)]


def table(target_id: str, *columns: tuple[str, list[str]]) -> dict:
    raw = {"table_id": target_id, "columns": [{"column_index": i, "column_name": n} for i, (n, _) in enumerate(columns)],
           "rows": [{"cells": [{"column_index": i, "text": texts[r]} for i, (_, texts) in enumerate(columns)]}
                    for r in range(len(columns[0][1]))]}
    return values.table_domain(raw)


def test_value_info_keeps_types_exact():
    assert values.value_info("1,234.50")["key"] == values.value_info("1234.5")["key"] == "NUMBER:1234.5"
    assert values.value_info("007")["key"] == "ID:007"
    assert values.value_info("2001-02-03")["kind"] == "DATE"
    assert values.value_info(" N/A ")["key"] is None
    assert values.value_info("Album 2 – Deluxe")["digits"] == ["2"]


@pytest.mark.parametrize("raw,expected", [
    ('"1999"', ("VALUE", "1999")), ("null", ("INSUFFICIENT_EVIDENCE", None)),
    ('```json\n"Paris"\n```', ("VALUE", "Paris")), ('{"value": 1}', ("PARSE_ERROR", None)),
    ('"n/a"', ("NORMALIZED_EMPTY", None))])
def test_parse_completion(raw, expected):
    assert values.parse_completion(raw) == expected


def prediction(unit, row, target, value, status="VALUE"):
    return {"unit_id": unit, "query_row_id": row, "target_id": target, "column_id": 0, "status": status, "value": value}


def test_bridges_turn_disagreeing_rows_into_conflicts():
    tables = {"t1": table("t1", ("Year", ["1990"])), "t2": table("t2", ("year", ["1991"]))}
    bridges = values.build_bridges("q", [prediction("a", 0, "t1", "1990"), prediction("b", 0, "t2", "1990.0"),
                                         prediction("c", 1, "t1", "1990"), prediction("d", 1, "t2", "1991")], tables)
    assert len(bridges) == 1 and bridges[0]["attribute"] == "year"
    slots = bridges[0]["slots"]
    assert slots[0]["status"] == "VALUE" and slots[0]["value_key"] == "NUMBER:1990"
    assert slots[1]["status"] == "CONFLICT" and slots[2]["status"] == "MISSING"
    assert bridges[0]["origin_targets_by_row"] == {"0": ["t1", "t2"]}


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_bridge_score_counts_rows_and_softens_text_only():
    tables = {"t": table("t", ("Label", ["Sony Music", "EMI", "x", "y", "z"]))}
    vectors = {"sony music": unit(1, 0), "sony": unit(1, 0.05), "emi": unit(0, 1), "e.m.i": unit(0.3, 1),
               "x": unit(1, 1), "y": unit(1, 1), "z": unit(1, 1)}
    preds = [prediction("a", 0, "t", "Sony"), prediction("b", 1, "t", "Sony"), prediction("c", 2, "t", "E.M.I")]
    bridges = values.build_bridges("q", preds, tables)
    score, winner = matching.bridge_scores(["t"], bridges, tables, matching.Matcher(vectors, 0.98))
    assert score["t"] == pytest.approx(2 / 5)  # "sony" softens to "sony music" twice; "e.m.i" stays below tau
    assert winner["t"]["matched_rows"] == 2


def test_numeric_tokens_block_semantic_match():
    vectors = {"season 1": unit(1, 0), "season 2": unit(1, 0)}
    matcher = matching.Matcher(vectors, 0.98)
    column = [values.value_info("Season 2")]
    assert not matcher.match(values.value_info("Season 1"), "c", column)


def test_visible_idf_discounts_values_every_candidate_contains():
    names = ["Ann", "Bob", "Cid", "Dee", "Eve"]
    countries = ["Peru", "Peru", "Chile", "Peru", "Chile"]
    query = rows(("Name", names), ("Country", countries))
    tables = {"common": table("common", ("Nation", ["Peru", "Chile"])),
              "rare": table("rare", ("Person", names)),
              "other": table("other", ("Nation", ["Peru", "Chile"]))}
    texts = visible.visible_texts(query, tables)  # orthogonal vectors: only exact matches count
    vectors = {t: np.eye(len(texts), dtype=np.float32)[i] for i, t in enumerate(sorted(texts))}
    vis_row, vis_idf = visible.visible_scores(["common", "rare", "other"], query, tables, matching.Matcher(vectors, 0.98))
    assert vis_row == {"common": 1.0, "rare": 1.0, "other": 1.0}
    # every country value occurs in 2 of 3 candidates, every name in 1
    assert vis_idf["common"] == vis_idf["other"] == pytest.approx(visible.idf_weight(2, 3))
    assert vis_idf["rare"] == pytest.approx(visible.idf_weight(1, 3))
    assert vis_idf["rare"] > vis_idf["common"]


def test_bridge_tier_always_precedes_visible_tier():
    candidates = ["a", "b", "c", "d"]
    order = visible.bidf_order(candidates, {"a": 0, "b": 0, "c": 0.2, "d": 0},
                               {"a": 1, "b": 0, "c": 0, "d": 0.4}, {"a": 0.9, "b": 0, "c": 0, "d": 0.4})
    assert order == ["c", "a", "d", "b"]
    assert matching.rrf_fuse(candidates, order, 60)[0] == "a"  # one-rank promotions only tie with Stage-1 rank 1


def test_plan_caps_columns_per_table_and_merges_identical_views():
    tables = {"t1": {"columns": [{"column_index": i, "column_name": f"c{i}"} for i in range(4)]},
              "t2": {"columns": [{"column_index": 0, "column_name": "c0"}]}}
    logits = {"t1": {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}, "t2": {0: 1.0}}
    views, selected = selector.build_plan("q", ["t1", "t2"], {"t1": 0.0, "t2": 0.0}, logits,
                                          {"t1": ["e1"], "t2": ["e1"]}, tables, branch_budget=10, per_table_cap=3)
    assert [(p["target_id"], p["column_id"]) for p in selected] == [("t2", 0), ("t1", 0), ("t1", 1), ("t1", 2)]
    c0 = next(v for v in views if v["column_name"] == "c0")
    assert [l["target_id"] for l in c0["donor_links"]] == ["t2", "t1"]  # same name + same bag -> one request


def recovery_query(rows_, assets):
    return {"query_id": "q", "rows": rows_, "assets": assets,
            "tables": {"t": table("t", ("Year", ["1990"]))}}


def test_requests_skip_observed_rows_and_retry_images_alone():
    query_rows = rows(("Title", ["A", "B", "C", "D", "E"]), ("Year", ["1990", "", "", "", ""]))
    assets = {"txt": {"asset_type": "text", "content": "..."}, "img": {"asset_type": "image", "local_path": "x"}}
    query = recovery_query(query_rows, assets)
    view = {"attribute": "year", "column_name": "Year", "evidence_ids": ["txt", "img"],
            "donor_links": [{"target_id": "t", "column_id": 0}]}
    packets, skips = recovery.packet_tasks(query, [view])
    assert [t["row"]["query_row_id"] for t in packets] == [1, 2, 3, 4] and skips[0]["row_id"] == 0
    answered = [{"query_row_id": 1, "target_id": "t", "column_id": 0, "status": "VALUE"}]
    singles, _ = recovery.singleton_tasks(query, [view], answered)
    assert [(t["row"]["query_row_id"], t["evidence_ids"]) for t in singles] == [(2, ["img"]), (3, ["img"]), (4, ["img"])]


def test_gate_rejects_text_about_other_rows_only():
    query_rows = rows(("Title", ["Abbey Road", "Let It Be", "Help Me", "Revolver", "Rubber Soul"]),
                      ("entity_url", [f"https://en.wikipedia.org/wiki/{t}" for t in
                                      ["Abbey_Road", "Let_It_Be", "Help_Me", "Revolver", "Rubber_Soul"]]))
    assets = {"e": {"asset_type": "text", "content": "Revolver was recorded in 1966."}}
    query = recovery_query(query_rows, assets)
    task = recovery.make_task(query, "PREFIX_PACKET", query_rows[0], "Year", ["e"], [])
    assert recovery.gate(task, query, 3000) == "OTHER_ROWS_ONLY"
    assert recovery.gate({**task, "row": query_rows[3]}, query, 3000) == "CURRENT_ENTITY_PRESENT"


def test_crop_box_abstains_on_diffuse_maps_and_crops_peaks():
    policy = json.loads((Path(__file__).resolve().parents[1] / "configs" / "mmdd_stage2_bidf.json").read_text())["crop"]
    flat = np.ones((16, 16))
    assert localizer.crop_box(flat, (640, 640), policy)["reason"] == "DIFFUSE_HEATMAP"
    yy, xx = np.mgrid[0:32, 0:32]
    peaked = 0.01 + np.exp(-((yy - 8) ** 2 + (xx - 22) ** 2) / 2)  # peak at pixel (440, 160)
    result = localizer.crop_box(peaked, (640, 640), policy)
    assert result["reason"] is None
    x0, y0, x1, y1 = result["tight_box"]
    assert x0 <= 440 < x1 and y0 <= 160 < y1
    cx0, cy0, cx1, cy1 = result["box"]
    assert cx0 <= x0 and cy0 <= y0 and cx1 >= x1 and cy1 >= y1
    assert (cx1 - cx0) * (cy1 - cy0) < policy["fallback_area_ratio"] * 640 * 640


def test_stage1_handoff_reads_c30_and_keeps_tail(tmp_path):
    results = [{"target_id": f"t{i}", "score": -float(i),
                "paths": [{"kind": "direct"}, {"kind": "evidence", "evidence_id": f"e{i}"}]} for i in range(50)]
    (tmp_path / "retrieval.dev.jsonl").write_text(json.dumps({"query_id": "q", "results": results}) + "\n")
    record = stage1.load_stage1(tmp_path, "dev", 30, 50)["q"]
    assert record["candidates"] == [f"t{i}" for i in range(30)] and len(record["ranking"]) == 50
    assert record["evidence"]["t3"] == ["e3"] and "t30" not in record["evidence"]


def test_recall_is_a_fraction_of_gold():
    m = metrics(["a", "x", "b"], {"a", "b", "c"}, [1, 3])
    assert m["R1"] == pytest.approx(1 / 3) and m["R3"] == pytest.approx(2 / 3)
    assert m["NDCG3"] == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3) + 1 / math.log2(4)))

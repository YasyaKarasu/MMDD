"""CPU reranking of each query's C30 from recovered bridges and visible IDF scores.

Writes ``scores/<split>.jsonl`` with every ranking policy, each a full Stage-1 depth list
(reranked C30 followed by the unchanged Stage-1 tail):

    STAGE1            the Stage-1 order
    BRIDGE_RRF60      recovered bridges only (RRF of the bridge order with Stage 1)
    VISIBLE_IDF_RRF60 visible IDF only, all bridge scores zero (what Stage 2 gives without evidence)
    BIDF_RRF60        bridges first, then visible IDF (the method)

A query whose recovery fell back to Stage 1 has no bridges, so its BIDF ranking equals its
VISIBLE_IDF ranking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .common import read_json, write_jsonl
from .matching import Matcher, TextVectors, bridge_order, bridge_scores, bridge_texts, rrf_fuse
from .stage1 import load_stage1
from .visible import bidf_ranking, visible_scores, visible_texts


def score_split(config: dict[str, Any], run: Path, split: str, catalog: Catalog, vectors: TextVectors) -> None:
    stage1 = load_stage1(Path(config["paths"]["stage1_handoff"]), split, config["candidate_scope"], config["output_depth"])
    population = read_json(run / "population" / f"{split}.json")

    def inputs():
        for query in population:
            record = stage1[query["query_id"]]
            yield (query["query_id"], record, read_json(run / "recovery" / split / f"{query['query_id']}.json"),
                   {t: catalog.get("target_domain", t) for t in record["candidates"]},
                   catalog.get("query_rows", query["query_id"]))

    texts: set[str] = set()
    for _, _, recovery, tables, rows in inputs():
        texts |= bridge_texts(recovery["bridges"], tables) | visible_texts(rows, tables)
    vectors.ensure(texts)

    k = config["fusion"]["rrf_constant"]
    matcher = Matcher(vectors, config["matching"]["tau_cosine"])
    out = []
    for query_id, record, recovery, tables, rows in inputs():
        matcher.clear()
        candidates, tail = record["candidates"], record["ranking"][len(record["candidates"]):]
        bridge, winners = bridge_scores(candidates, recovery["bridges"], tables, matcher)
        vis_row, vis_idf = visible_scores(candidates, rows, tables, matcher)
        zero = {t: 0.0 for t in candidates}
        out.append({
            "query_id": query_id, "recovery_status": recovery["status"],
            "recovered_rows": sum(s["status"] == "VALUE" for b in recovery["bridges"] for s in b["slots"]),
            "bridge_scores": bridge, "bridge_winners": winners, "vis_row": vis_row, "vis_idf": vis_idf,
            "rankings": {
                "STAGE1": record["ranking"],
                "BRIDGE_RRF60": rrf_fuse(candidates, bridge_order(candidates, bridge), k) + tail,
                "VISIBLE_IDF_RRF60": bidf_ranking(candidates, zero, vis_row, vis_idf, k) + tail,
                "BIDF_RRF60": bidf_ranking(candidates, bridge, vis_row, vis_idf, k) + tail,
            }})
    write_jsonl(run / "scores" / f"{split}.jsonl", out)
    print(json.dumps({"split": split, "queries": len(out),
                      "with_bridge_score": sum(any(v > 0 for v in r["bridge_scores"].values()) for r in out),
                      "with_visible_score": sum(any(v > 0 for v in r["vis_idf"].values()) for r in out)}), flush=True)


def run_scoring(config: dict[str, Any], run: Path) -> None:
    catalog = Catalog(run)
    vectors = TextVectors(run / "matching", Path(config["paths"]["minilm_model"]), config["matching"],
                          config["cpu_threads"])
    for split in ("dev", "test"):
        score_split(config, run, split, catalog, vectors)

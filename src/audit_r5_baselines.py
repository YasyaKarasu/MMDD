"""Compare Q+ baselines and replay the strongest baseline's raw table rankings."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_r5_artifacts import paired, rank_metrics, read_csv, records, write_csv


def audit(root: Path, baseline: Path, output: Path) -> None:
    materialized = baseline / "materialized"
    manifest = json.loads((materialized / "manifest.json").read_text())
    mappings = json.loads((materialized / "mappings/mappings.json").read_text())
    run = root / "work/stage2_entitables_r5_sup_crop_on"
    population = {r["query_id"]: r for r in json.loads((run / "population/test.json").read_text())}
    assert set(mappings["query_id_to_file"]) == set(population)
    db = sqlite3.connect(f"file:{run / 'catalog.sqlite'}?mode=ro", uri=True)
    targets = {r[0] for r in db.execute("SELECT id FROM objects WHERE kind='target'")}
    db.close()
    assert set(mappings["table_id_to_file"]) == targets
    gold = defaultdict(set)
    for row in records(Path(manifest["dataset_root"]) / "qrels.jsonl"):
        if row["split"] == "test" and row["rel"] > 0:
            gold[row["query_table_id"]].add(row["target_table_id"])
    s1_gold = defaultdict(set)
    for r in records(root / "work/stage1_entitables_r5_s13_sup_rerank/eval_labels/test/qrels.jsonl"):
        if r["rel"] > 0:
            s1_gold[r["query_id"]].add(r["target_id"])
    assert gold == s1_gold
    summary = []
    for path in sorted((baseline / "results").glob('*/*table_level_metrics.json')):
        data = json.loads(path.read_text())
        for pop in ("total", "implicit", "explicit"):
            summary.append({"method": path.parent.name, "population": pop,
                            **{f"{metric}{k}": data[key][pop][str(k)]
                               for metric, key in (("R", "recall_at_ks_by_reason"), ("NDCG", "ndcg_at_ks_by_reason"))
                               for k in (5, 10, 15, 20)}})
    by_column = defaultdict(list)
    with (baseline / "results/mosaicjoin/query_results/all_query_results.csv").open() as handle:
        for r in csv.DictReader(handle):
            by_column[r["query_table"].strip(), r["query_column"].strip()].append(
                (float(r.get("similarity_score", 0)), r["candidate_table"].strip()))
    qnames = {v: k for k, v in mappings["query_id_to_file"].items()}
    tnames = {v: k for k, v in mappings["table_id_to_file"].items()}
    best = defaultdict(dict)
    def canonical(name: str) -> str:
        return name if name.endswith(".csv") else name + ".csv"
    for (qname, _column), candidates in by_column.items():
        qid = qnames.get(canonical(qname))
        if qid is None:
            continue
        rank = 0
        for _score, tname in sorted(candidates, key=lambda x: x[0], reverse=True):
            tid = tnames.get(canonical(tname))
            if tid is None:
                continue
            rank += 1
            best[qid][tid] = min(rank, best[qid].get(tid, rank))
    ours = {r["query_id"]: r for r in read_csv(run / "evaluation/PER_QUERY.csv")
            if r["split"] == "test" and r["policy"] == "BIDF_RRF60"}
    replay = []
    for qid in sorted(population):
        order = sorted(best[qid], key=best[qid].get)
        replay.append({"query_id": qid, "kind": ours[qid]["kind"],
                       "source_group": population[qid]["source_group"], **rank_metrics(order, gold[qid])})
    contrasts, max_error = [], 0.0
    for pop in ("total", "implicit", "explicit"):
        selected = [r for r in replay if pop == "total" or r["kind"] == pop]
        saved = next(r for r in summary if r["method"] == "mosaicjoin" and r["population"] == pop)
        for key in ("R5", "NDCG5", "R10", "NDCG10", "R15", "NDCG15", "R20", "NDCG20"):
            max_error = max(max_error, abs(float(np.mean([r[key] for r in selected])) - saved[key]))
            contrasts.append({"population": pop, "metric": key, "comparison": "R5_BIDF_minus_Qplus_MosaicJoin",
                **paired([r["source_group"] for r in selected], [float(ours[r["query_id"]][key]) - r[key] for r in selected])})
    assert max_error < 1e-12
    write_csv(output / "BASELINES.csv", summary)
    write_csv(output / "MOSAIC_REPLAY.csv", replay)
    write_csv(output / "R5_MINUS_MOSAIC.csv", contrasts)
    info = {"queries": len(population), "targets": len(targets), "positive_pairs": sum(map(len, gold.values())),
            "same_query_target_and_gold_sets": True, "mosaic_summary_max_error": max_error,
            "scope": "Q+ baseline system comparison, not a common-candidate component ablation"}
    (output / "BASELINE_AUDIT.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.baseline, args.output)

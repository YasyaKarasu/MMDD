"""Independent JSONL qrels + rankings evaluator. No imports from training code.

Reads the exported ``eval_labels/<split>/{queries,qrels}.jsonl`` and a ``rankings.*.jsonl.gz``
file and recomputes query-macro target recall from raw IDs. ``pipeline._independent_verify``
compares its numbers with the training-side metrics; it can also run standalone:

    python -m mmdd_stage1.independent_metrics --queries Q --qrels R --rankings F --out DIR
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

CUTOFFS = (10, 20, 30, 40, 50)
SEGMENTS = ("overall", "implicit", "explicit", "mixed", "unknown")
FIELDS = ("R10", "R20", "R30", "R40", "R50", "CR_pool", "QueryHitRate_pool", "Oracle10")


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def evaluate(queries: Path, qrels: Path, rankings: Path, out: Path, allow_subset: bool = False) -> dict:
    meta = {}
    for row in rows(queries):
        query_id = str(row["query_id"])
        if query_id in meta:
            raise ValueError(f"duplicate query {query_id}")
        if not row.get("source_group"):
            raise ValueError(f"missing source_group {query_id}")
        meta[query_id] = row
    gold = defaultdict(set)
    for row in rows(qrels):
        query_id = str(row["query_id"])
        if query_id not in meta:
            raise ValueError(f"qrels query absent from metadata: {query_id}")
        if float(row["rel"]) > 0:
            gold[query_id].add(str(row["target_id"]))

    data, seen = [], set()
    for row in rows(rankings):
        query_id, ids = str(row["query_id"]), [str(x) for x in row["target_ids"]]
        if query_id not in meta or query_id in seen:
            raise ValueError(f"unknown/duplicate ranking query {query_id}")
        seen.add(query_id)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate ranked ID: {query_id}")
        if "scores" in row:
            scores = row["scores"]
            if len(scores) != len(ids) or not all(math.isfinite(float(x)) for x in scores):
                raise ValueError(f"invalid scores {query_id}")
            expected = sorted(zip(ids, scores), key=lambda x: (-x[1], x[0].encode()))
            if [i for i, _ in expected] != ids:
                raise ValueError(f"not sorted by score/tie rule {query_id}")
        targets = gold[query_id]
        if not targets:
            continue
        record = {
            "query_id": query_id, "source_group": str(meta[query_id]["source_group"]),
            "query_kind": meta[query_id].get("query_kind", "unknown"), "gold_count": len(targets), "pool_size": len(ids),
        }
        for k in CUTOFFS:
            hits = len(set(ids[:k]) & targets)
            record[f"hits{k}"], record[f"R{k}"] = hits, hits / len(targets)
        recovered = len(set(ids) & targets)
        record["gold_in_pool"], record["CR_pool"] = recovered, recovered / len(targets)
        record["QueryHitRate_pool"], record["Oracle10"] = int(recovered > 0), min(10, recovered) / len(targets)
        data.append(record)

    missing = {q for q in meta if gold[q]} - seen
    if missing and not allow_subset:
        raise ValueError(f"{len(missing)} positive queries have no ranking; explicit --allow-subset required for registered probe")
    if not data:
        raise ValueError("no evaluable ranked queries")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    result = {
        "source": "independent_raw_qrels_and_rankings", "n_metadata_queries": len(meta),
        "n_scored_queries": len(data), "subset": bool(missing), "missing_queries": sorted(missing),
        "zero_gold_exclusions": sorted(q for q in meta if not gold[q]), "segments": {},
    }
    for kind in SEGMENTS:
        part = data if kind == "overall" else [r for r in data if r["query_kind"] == kind]
        result["segments"][kind] = {
            "n": len(part), **{f: sum(r[f] for r in part) / len(part) if part else None for f in FIELDS},
        }
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "qrels", "rankings", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.queries, args.qrels, args.rankings, args.out, args.allow_subset), ensure_ascii=False, indent=2))

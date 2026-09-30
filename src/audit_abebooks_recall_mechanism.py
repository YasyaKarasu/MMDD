"""Separate implicit-label recall from witnessed, non-visible attribute hits.

This is a post-hoc diagnostic only: it never changes training examples, qrels,
rankings, or model selection. Literal visibility is not semantic verification.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import unicodedata
from pathlib import Path

from mmdd_dataset.abebooks_ablation import read_rows


def words(value: str) -> str:
    return " " + " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold())) + " "


def query_mechanism(query: dict, gold: set[str], facts: list[dict], pool: dict,
                    canonical: dict[str, str]) -> dict:
    """Audit every known recovery fact, without treating unlabelled paths as wrong."""
    rows = [words(" ".join(c["text"] for c in row["cells"])) for row in query["rows"]]
    entire_query = " ".join(rows)
    details = []
    for fact in facts:
        attr = fact["recovered_attribute"]
        value = words(attr["value"])
        eid = canonical[fact["evidence"]["asset_id"]]
        tid = fact["target_table_id"]
        details.append({"target_id": tid, "query_row": fact["query_row_id"],
            "attribute": attr["column_name"], "value": attr["value"], "evidence_id": eid,
            "visible_in_row": value.strip() != "" and value in rows[fact["query_row_id"]],
            "visible_in_query": value.strip() != "" and value in entire_query,
            "gold_witness_retained": eid in pool["retained_bags"].get(tid, [])})
    direct, rrf = pool["D100_ANN"], pool["C150"]
    witnessed = {f["target_id"] for f in details
                 if not f["visible_in_query"] and f["gold_witness_retained"]}
    novel = gold & set(rrf[:10]) - set(direct[:10])
    return {"query_id": query["table_id"], "split": query["split"],
        "RRF_recall10": len(gold & set(rrf[:10])) / len(gold),
        "Direct_recall10": len(gold & set(direct[:10])) / len(gold),
        "any_known_value_visible": any(f["visible_in_query"] for f in details),
        "witnessed_nonvisible_recall10": len(gold & set(rrf[:10]) & witnessed) / len(gold),
        "new_witnessed_nonvisible_recall10": len(novel & witnessed) / len(gold),
        "gold_ranks": {t: {"Direct": direct.index(t) + 1 if t in direct else None,
                             "RRF": rrf.index(t) + 1 if t in rrf else None} for t in sorted(gold)},
        "facts": details}


def audit(root: Path, splits: list[str]) -> dict:
    run, dataset = root / "main", root / "dataset_view"
    queries = {q["table_id"]: q for q in read_rows(dataset / "query_tables/part-00000.jsonl")}
    relations = read_rows(dataset / "qrels.jsonl")
    recoveries = read_rows(dataset / "evidence_recoveries/part-00000.jsonl")
    with gzip.open(run / "CONTENT_ALIASES.jsonl.gz", "rt") as handle:
        aliases = {r["asset_id"]: r["canonical_evidence_id"] for r in map(json.loads, handle)}
    records = []
    for split in splits:
        with gzip.open(run / "eval" / split / "selected_kd/pools.jsonl.gz", "rt") as handle:
            pools = {p["query_id"]: p for p in map(json.loads, handle)}
        for qid, query in queries.items():
            gold = {r["target_table_id"] for r in relations if r["query_table_id"] == qid
                    and r["reason"] == "model_recoverable_join_column" and r["rel"] > 0}
            if query["split"] != split or not gold:
                continue
            facts = [r for r in recoveries if r["query_table_id"] == qid]
            records.append(query_mechanism(query, gold, facts, pools[qid], aliases))
    summary = {}
    for split in splits:
        selected = [r for r in records if r["split"] == split]
        summary[split] = {"implicit_queries": len(selected),
            "queries_with_visible_known_values": sum(r["any_known_value_visible"] for r in selected),
            **{metric: sum(r[metric] for r in selected) / len(selected) for metric in
               ("RRF_recall10", "Direct_recall10", "witnessed_nonvisible_recall10", "new_witnessed_nonvisible_recall10")}}
    result = {"scope": "Selected KD, all implicit queries; diagnostic only, unchanged training and evaluation labels",
        "limits": "Exact normalized value visibility and existing witness coverage; neither semantic absence nor new value generation/join verification is established. Unlabelled paths may be valid.",
        "summary": summary, "queries": records}
    (root / "MECHANISM_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("dev", "test"), default=["dev", "test"])
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root.resolve(), args.splits)["summary"], indent=2))


if __name__ == "__main__":
    main()

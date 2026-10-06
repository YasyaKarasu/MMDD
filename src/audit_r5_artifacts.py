"""Independently recompute frozen R4/R5 rankings and paired source-group intervals.

Run from an isolated cwd. Reads only named experiment artifacts; no model/client imports.
Outputs small audit tables, never modifies the original runs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_metrics(order: list[str], gold: set[str]) -> dict[str, float]:
    result = {}
    for k in (1, 5, 10, 15, 20, 30, 50):
        hits = [i for i, t in enumerate(order[:k]) if t in gold]
        result[f"R{k}"] = len(hits) / len(gold)
        result[f"P{k}"] = len(hits) / k
        result[f"NDCG{k}"] = sum(1 / math.log2(i + 2) for i in hits) / sum(
            1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return result


def paired(groups: list[str], deltas: list[float]) -> dict:
    grouped = defaultdict(list)
    for group, delta in zip(groups, deltas):
        grouped[group].append(delta)
    sums = np.array([(sum(grouped[g]), len(grouped[g])) for g in sorted(grouped)])
    rng = np.random.default_rng(20260925)
    means = []
    for _ in range(40):
        selected = sums[rng.integers(len(sums), size=(250, len(sums)))].sum(axis=1)
        means.extend(selected[:, 0] / selected[:, 1])
    d = np.asarray(deltas)
    return {"delta": float(d.mean()), "CI_low": float(np.quantile(means, .025)),
            "CI_high": float(np.quantile(means, .975)), "W": int((d > 1e-12).sum()),
            "L": int((d < -1e-12).sum()), "T": int((abs(d) <= 1e-12).sum()),
            "queries": len(d), "groups": len(grouped)}


def audit(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    s1 = root / "work/stage1_entitables_r5_s13_sup_rerank"
    runs = {n: root / f"work/stage2_entitables_r{n}{suffix}_crop_on"
            for n, suffix in ((4, ""), (5, "_sup"))}
    configs = {n: json.loads((r / "config.json").read_text()) for n, r in runs.items()}
    populations, scores, calculated, gold = {}, {}, {}, {}
    checks, summaries, recovery_rows, pool_rows = [], [], [], []
    for split in ("dev", "test"):
        g = defaultdict(set)
        for row in records(s1 / f"eval_labels/{split}/qrels.jsonl"):
            if row["rel"] > 0:
                g[row["query_id"]].add(row["target_id"])
        gold[split] = g
    for n, run in runs.items():
        per_query = {(r["split"], r["query_id"], r["policy"]): r
                     for r in read_csv(run / "evaluation/PER_QUERY.csv")}
        populations[n], scores[n], calculated[n] = {}, {}, {}
        max_error = 0.0
        for split in ("dev", "test"):
            pop = json.loads((run / f"population/{split}.json").read_text())
            populations[n][split] = {r["query_id"]: r for r in pop}
            score_rows = list(records(run / f"scores/{split}.jsonl"))
            scores[n][split] = {r["query_id"]: r for r in score_rows}
            assert len(score_rows) == len(pop) == len(scores[n][split])
            assert scores[n][split].keys() == populations[n][split].keys() == gold[split].keys()
            handoff = Path(configs[n]["paths"]["stage1_handoff"])
            manifest = json.loads((handoff / "EXPORT_MANIFEST.json").read_text())
            digest = hashlib.sha256((handoff / f"retrieval.{split}.jsonl").read_bytes()).hexdigest()
            assert digest == manifest["retrievals"][split]["sha256"]
            retrieval = {r["query_id"]: r for r in records(handoff / f"retrieval.{split}.jsonl")}
            for q, record in scores[n][split].items():
                original = record["rankings"]["STAGE1"]
                assert original == [r["target_id"] for r in retrieval[q]["results"]]
                for policy, order in record["rankings"].items():
                    assert len(order) == len(set(order)) == 50
                    assert set(order[:30]) == set(original[:30]) and order[30:] == original[30:]
                    value = rank_metrics(order, gold[split][q])
                    saved = per_query[split, q, policy]
                    max_error = max(max_error, max(abs(v - float(saved[k])) for k, v in value.items()))
                    calculated[n][split, q, policy] = {**saved, **value}
                rec = json.loads((run / f"recovery/{split}/{q}.json").read_text())
                tasks = rec.get("tasks", [])
                slots = [s for b in rec["bridges"] for s in b["slots"]]
                crop = [c for t in tasks for c in t.get("crops", [])]
                recovery_rows.append({"run": n, "split": split, "query_id": q,
                    "kind": per_query[split, q, "STAGE1"]["kind"], "status": rec["status"],
                    "model_inputs": rec.get("model_inputs", 0), "seconds": rec.get("seconds", 0),
                    "value_slots": sum(s["status"] == "VALUE" for s in slots),
                    "unique_rows_with_value": len({s["row_id"] for s in slots if s["status"] == "VALUE"}),
                    "bridge_attributes": len(rec["bridges"]), "tasks": len(tasks),
                    "value_tasks": sum(t["status"] == "VALUE" for t in tasks),
                    "crop_records": len(crop), "accepted_crop_records": sum(c["box"] is not None for c in crop),
                    "positive_bridge_targets": sum(v > 0 for v in record["bridge_scores"].values()),
                    "gold_bridge_targets": sum(record["bridge_scores"].get(t, 0) > 0 for t in gold[split][q])})
        assert max_error < 1e-12
        summary_error = 0.0
        for saved in read_csv(run / "evaluation/METRICS.csv"):
            selected = [r for (s, q, p), r in calculated[n].items()
                        if s == saved["split"] and p == saved["policy"]
                        and (saved["population"] == "overall" or r["kind"] == saved["population"])]
            values = {k: float(np.mean([r[k] for r in selected]))
                      for k in selected[0] if k.startswith(("R", "P", "NDCG"))}
            summary_error = max(summary_error, max(abs(v - float(saved[k])) for k, v in values.items()))
            summaries.append({"run": n, **saved})
        contrast_error = 0.0
        for row in read_csv(run / "evaluation/CONTRASTS.csv"):
            left = {q: r for (s, q, p), r in calculated[n].items() if s == row["split"] and p == row["method"]
                    and (row["population"] == "overall" or r["kind"] == row["population"])}
            ids = sorted(left)
            recomputed = paired([left[q]["source_group"] for q in ids], [left[q][row["metric"]] -
                calculated[n][row["split"], q, row["reference"]][row["metric"]] for q in ids])
            contrast_error = max(contrast_error, max(abs(recomputed[k] - float(row[k]))
                                                    for k in ("delta", "CI_low", "CI_high", "W", "L", "T")))
        checks.append({"run": n, "per_query_max_error": max_error,
                       "summary_max_error": summary_error, "contrast_max_error": contrast_error})
    contrasts = []
    for split in ("dev", "test"):
        assert populations[4][split] == populations[5][split]
        for q in scores[5][split]:
            a, b = [scores[n][split][q]["rankings"]["STAGE1"] for n in (4, 5)]
            pool_rows.append({"split": split, "query_id": q, "C30_same": set(a[:30]) == set(b[:30]),
                              "C30_jaccard": len(set(a[:30]) & set(b[:30])) / len(set(a[:30]) | set(b[:30])),
                              "top10_same_order": a[:10] == b[:10]})
        for pop in ("overall", "implicit", "explicit"):
            ids = sorted(q for q in populations[5][split] if pop == "overall" or calculated[5][split, q, "STAGE1"]["kind"] == pop)
            groups = [populations[5][split][q]["source_group"] for q in ids]
            for metric in ("R5", "NDCG5", "R10", "NDCG10", "R20", "NDCG20", "R30", "R50"):
                for policy in ("STAGE1", "BRIDGE_RRF60", "VISIBLE_IDF_RRF60", "BIDF_RRF60", "evidence_DiD", "bridge_vs_stage1_DiD"):
                    def value(n: int, q: str) -> float:
                        if policy in ("evidence_DiD", "bridge_vs_stage1_DiD"):
                            ref = "VISIBLE_IDF_RRF60" if policy == "evidence_DiD" else "STAGE1"
                            return calculated[n][split, q, "BIDF_RRF60"][metric] - calculated[n][split, q, ref][metric]
                        return calculated[n][split, q, policy][metric]
                    contrasts.append({"split": split, "population": pop, "comparison": f"R5_minus_R4:{policy}",
                                      "metric": metric, **paired(groups, [value(5, q) - value(4, q) for q in ids])})
    write_csv(output / "RECOMPUTED_METRICS.csv", summaries)
    write_csv(output / "R5_MINUS_R4.csv", contrasts)
    write_csv(output / "RECOVERY_COUNTS.csv", recovery_rows)
    write_csv(output / "POOL_COMPARISON.csv", pool_rows)
    audit_info = {"checks": checks, "bootstrap": {"replicates": 10000, "seed": 20260925,
                 "unit": "source_group", "aggregation": "query_weighted"}, "pools": {}, "recovery": {}}
    for split in ("dev", "test"):
        ps = [r for r in pool_rows if r["split"] == split]
        audit_info["pools"][split] = {"queries": len(ps), "identical_C30": sum(r["C30_same"] for r in ps),
                                    "mean_C30_jaccard": float(np.mean([r["C30_jaccard"] for r in ps]))}
        for n in (4, 5):
            for pop in ("overall", "implicit", "explicit"):
                rs = [r for r in recovery_rows if r["run"] == n and r["split"] == split and (pop == "overall" or r["kind"] == pop)]
                audit_info["recovery"][f"R{n}/{split}/{pop}"] = {"queries": len(rs),
                    "status": dict(Counter(r["status"] for r in rs)),
                    "queries_with_value": sum(r["value_slots"] > 0 for r in rs),
                    "queries_with_bridge_score": sum(r["positive_bridge_targets"] > 0 for r in rs),
                    "queries_with_gold_bridge_score": sum(r["gold_bridge_targets"] > 0 for r in rs),
                    **{k: sum(r[k] for r in rs) for k in ("value_slots", "unique_rows_with_value", "model_inputs", "seconds", "tasks", "value_tasks", "crop_records", "accepted_crop_records")}}
    (output / "AUDIT.json").write_text(json.dumps(audit_info, indent=2) + "\n")
    print(json.dumps({"output": str(output), "checks": checks, "pools": audit_info["pools"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)

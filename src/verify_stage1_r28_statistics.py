"""Independently verify final summary values and primary source-bootstrap contrasts."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from prepare_stage1_r27 import record, rows, write_json
from prepare_stage1_r28 import OUT


def main() -> None:
    assert not json.loads((OUT / "ANALYSIS_STATUS.json").read_text())["missing"]
    dimensions = ("section", "arm", "seed", "epoch", "budget", "view", "condition")
    groups = {}
    paired = defaultdict(dict)
    population = {}
    row_count = 0
    for row in rows(OUT / "statistics/per_query.jsonl.gz"):
        row_count += 1
        q = row["query_id"]
        meta = (row["source_table_id"], row["query_kind"])
        assert population.setdefault(q, meta) == meta
        key = tuple(str(row[d]) if d != "epoch" else str(float(row[d])) for d in dimensions)
        for kind in ("overall", row["query_kind"]):
            group = groups.setdefault((*key, kind), {"ids": set(), "sums": defaultdict(float)})
            assert q not in group["ids"], (key, q)
            group["ids"].add(q)
            for metric, value in row["metrics"].items():
                group["sums"][metric] += value
        if (row["section"] == "teacher" and row["budget"] == "Full-U") or row["section"] == "student":
            for metric in ("R@10", "RawRecall", "EO_STRICT_hits@10"):
                if metric in row["metrics"]:
                    contrast = (row["section"], row["arm"], row["epoch"], row["view"], row["condition"], metric)
                    paired[contrast][row["seed"], q] = row["metrics"][metric]
        if row["section"] == "teacher":
            for k in (10, 20, 50):
                m = row["metrics"]
                assert abs(m[f"strict_contribution@{k}"] + m[f"non_strict_contribution@{k}"] - m[f"R@{k}"]) < 1e-12
    assert len(population) == 1198
    for key, group in list(groups.items()):
        if key[2] != "13":
            continue
        other = groups[(*key[:2], "29", *key[3:])]
        assert group["ids"] == other["ids"]
        groups[(*key[:2], "13+29", *key[3:])] = {
            "ids": group["ids"], "sums": {m: (v + other["sums"][m]) / 2 for m, v in group["sums"].items()}}
    observed = set()
    with (OUT / "statistics/main_table.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[d] if d != "epoch" else str(float(row[d])) for d in dimensions) + (row["kind"],)
            metric = row["metric"]
            assert (key, metric) not in observed
            observed.add((key, metric))
            group = groups[key]
            n = len(group["ids"])
            assert int(row["queries"]) == n
            assert abs(float(row["value"]) - group["sums"][metric] / n) < 1e-11
            if row["total"]:
                assert abs(float(row["total"]) - group["sums"][metric]) < 1e-8
    assert observed == {(key, m) for key, group in groups.items() for m in group["sums"]}
    checked = []
    for result in rows(OUT / "statistics/source_group_bootstrap.jsonl"):
        assert result["replicates"] == 10000
        assert result["wins"] + result["losses"] + result["ties"] == result["queries"]
        left_key, right_key = tuple(result["left"]), tuple(result["right"])
        if result["seeds"] != [13, 29] or result["kind"] != "overall":
            continue
        if result["comparison"] not in ("Path vs Edge", "COV vs Edge", "COV vs LSE", "Epoch5 vs Epoch1", "Real vs Shuffled", "Epoch5 vs fixed T0"):
            continue
        primary = (left_key[0] == "teacher" and left_key[3] in ("D", "E-LSE", "E-COV")
                   and left_key[-1] in ("R@10", "EO_STRICT_hits@10")) or (
                   left_key[0] == "student" and (left_key[3], left_key[-1]) in (("U", "RawRecall"), ("U_OFFLINE_T0", "R@10")))
        if not primary:
            continue
        qids = sorted(population)
        left, right = paired[left_key], paired[right_key]
        delta = np.asarray([sum(left[(s, q) if (s, q) in left else (0, q)]
                                  - right[(s, q) if (s, q) in right else (0, q)]
                                  for s in (13, 29)) / 2 for q in qids])
        sources = sorted({population[q][0] for q in qids})
        by_source = [[delta[i] for i, q in enumerate(qids) if population[q][0] == source] for source in sources]
        sums = np.array([sum(values) for values in by_source])
        counts = np.array([len(values) for values in by_source])
        # Independently form the same registered source draws, without using the analysis helper.
        draws = np.random.default_rng(280915).integers(len(sources), size=(10000, len(sources)))
        samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        interval = np.quantile(samples, [.025, .975])
        assert np.allclose(interval, result["bootstrap_95ci"], atol=1e-12, rtol=0)
        assert abs(float(delta.mean()) - result["mean_delta"]) < 1e-12
        assert int((delta > 1e-12).sum()) == result["wins"]
        assert int((delta < -1e-12).sum()) == result["losses"]
        checked.append({"comparison": result["comparison"], "left": result["left"],
                        "mean_delta": result["mean_delta"], "bootstrap_95ci": interval.tolist()})
    assert checked
    receipt = {"status": "pass", "code": record(Path(__file__)), "per_query_rows": row_count, "summary_cells": len(observed),
               "independently_recomputed_primary_comparisons": checked,
               "inputs": [record(OUT / "statistics" / p) for p in ("per_query.jsonl.gz", "main_table.csv", "source_group_bootstrap.jsonl")]}
    write_json(OUT / "INDEPENDENT_STATISTICS_AUDIT.json", receipt)
    print(json.dumps({"status": "pass", "per_query_rows": row_count, "summary_cells": len(observed), "bootstrap_checks": len(checked)}))


if __name__ == "__main__":
    main()

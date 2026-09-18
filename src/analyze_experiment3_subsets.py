#!/usr/bin/env python
"""Where did Experiment 3's gain come from?

Joins the Experiment-1 witness-retention outcome of every positive target to
each arm's frozen-support ranking, so the B-minus-A gain can be attributed:

  * `retained`            -- a verified witness is inside the budget-4 bag
  * `not_in_pre_retention`-- a verified witness exists but retrieval missed it
  * `annotation_unknown`  -- no witness label for this target at all

Reported per target (GT-in-TopK rate), not query-macro, so the cohorts do not
overlap or reweight each other.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "work/witness_diagnostic_20260916"
EVAL = IN / "experiment3/eval"
OUT = IN / "experiment3"
KS = (10, 20, 50)
ORDER = ["retained", "lost_at_budget4", "lost_at_top20", "not_in_pre_retention", "annotation_unknown"]


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    outcome: dict[tuple[str, str, str], str] = {}
    with (IN / "per_target_classification.csv").open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            outcome[(record["endpoint"], record["query_id"], record["target_id"])] = record["outcome"]

    labels = ["parent", "armA", "armB", "armA_s29", "armB_s29", "armA_b4", "armB_b4"]
    labels = [label for label in labels if (EVAL / f"{label}_rankings.json").is_file()]
    rankings = {
        label: json.loads((EVAL / f"{label}_rankings.json").read_text()) for label in labels
    }

    hits: dict[str, Counter] = defaultdict(Counter)
    for (endpoint, query_id, target_id), result in outcome.items():
        key = f"{endpoint}/{query_id}"
        for label in labels:
            ranking = rankings[label].get(key)
            if ranking is None:
                continue
            hits[result][f"{label}|total"] += 1
            for k in KS:
                if target_id in set(ranking[:k]):
                    hits[result][f"{label}|top{k}"] += 1

    table = []
    for result in ORDER:
        counter = hits.get(result)
        if not counter:
            continue
        row = {"outcome": result, "targets": counter["parent|total"]}
        for label in labels:
            for k in KS:
                total = counter[f"{label}|total"]
                row[f"{label}_top{k}_pct"] = (
                    round(100 * counter[f"{label}|top{k}"] / total, 3) if total else None
                )
        table.append(row)
    with (OUT / "experiment3_subset_gain.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    missed = sum(len(hits[result]) for result in hits) == 0
    report = {
        "unit": "per positive target in the frozen C100, GT-in-TopK rate",
        "subsets": table,
        "note": (
            "Cohorts are the Experiment-1 witness outcomes; they are disjoint and cover "
            "every C100 positive."
        ),
        "empty": missed,
    }
    (OUT / "EXPERIMENT3_SUBSET_GAIN.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in table:
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

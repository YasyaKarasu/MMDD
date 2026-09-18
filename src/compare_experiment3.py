#!/usr/bin/env python
"""Compare the Experiment-3 arms across training-seed replications.

Reads the per-arm evaluation outputs and reports, on the identical frozen
support P:

  * B vs A per replication  -- the single-factor effect of adding L_target_path
  * B vs A averaged over training seeds (average within query first, then
    resample whole source_table_id groups) -- the headline number
  * B vs parent and B vs QT-on-P for the same seed set

QT-on-P is recomputed here from the frozen QT scores restricted to P; it is not
copied from any earlier report.  Data schedule (edge batch order, bag order) is
protocol-fixed and identical across replications, so a seed replication varies
only model-side randomness.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RERANK = ROOT / "work/final_rerank_20260916/FINAL_RERANK"
EVAL = ROOT / "work/witness_diagnostic_20260916/experiment3/eval"
OUT = ROOT / "work/witness_diagnostic_20260916/experiment3"
KS = (10, 20, 50)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 260916

# training-seed replications: tag -> (arm A label, arm B label)
REPLICATIONS = {"seed13": ("armA", "armB"), "seed29": ("armA_s29", "armB_s29")}
# candidate-source replays: same seed, different frozen Student train-side candidates
CANDIDATE_REPLAYS = {"b4": ("armA_b4", "armB_b4")}


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def recall_at(ranking: list[str], positives: list[str], k: int) -> float:
    truth = set(positives)
    unique = list(dict.fromkeys(ranking))
    return len(truth & set(unique[:k])) / len(truth)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="parent")
    args = parser.parse_args()

    membership = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "candidates/path_membership.jsonl.gz")
    }
    qt = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "rankings/QT.jsonl.gz")
    }
    qt_scores = {
        (record["endpoint"], record["query_id"]): record["scores"]
        for record in rows(RERANK / "scores/QT_scores.jsonl.gz")
    }

    support: dict[tuple[str, str], set[str]] = {}
    source_group: dict[tuple[str, str], str] = {}
    positives: dict[tuple[str, str], list[str]] = {}
    kinds: dict[tuple[str, str], str] = {}
    for key, record in membership.items():
        support[key] = {
            str(target["target_id"])
            for target in record["targets"]
            if target.get("retained_paths")
        }
        source_group[key] = str(record["source_table_id"])
        positives[key] = [str(value) for value in qt[key]["positive_target_ids"]]
        kinds[key] = str(qt[key]["query_kind"])

    labels = [args.baseline, "QT-on-P"]
    for arm_a, arm_b in REPLICATIONS.values():
        labels.extend([arm_a, arm_b])
    for arm_a, arm_b in CANDIDATE_REPLAYS.values():
        labels.extend([arm_a, arm_b])
    labels = [l for l in labels if (EVAL / f"{l}_rankings.json").is_file()]

    per_query: dict[str, dict[tuple[str, str], dict[int, float]]] = {}
    for label in labels:
        if label == "QT-on-P":
            continue
        rankings = json.loads((EVAL / f"{label}_rankings.json").read_text())
        per_query[label] = {
            key: {
                k: recall_at(rankings[f"{key[0]}/{key[1]}"], positives[key], k) for k in KS
            }
            for key in membership
            if f"{key[0]}/{key[1]}" in rankings
        }

    qt_on_p: dict[tuple[str, str], dict[int, float]] = {}
    for key in membership:
        scores = qt_scores[key]
        ranking = sorted(
            (target for target in support[key] if target in scores),
            key=lambda target: (-float(scores[target]), target),
        )
        qt_on_p[key] = {k: recall_at(ranking, positives[key], k) for k in KS}
    per_query["QT-on-P"] = qt_on_p

    keys = sorted(per_query[args.baseline])

    def macro(label: str, bucket: str) -> dict[int, float]:
        selected = [key for key in keys if bucket == "overall" or kinds[key] == bucket]
        return {
            k: sum(per_query[label][key][k] for key in selected) / len(selected) for k in KS
        }

    table_rows = []
    for bucket in ("overall", "implicit", "explicit"):
        for label in labels:
            values = macro(label, bucket)
            table_rows.append(
                {"bucket": bucket, "view": label,
                 **{f"R@{k}": round(100 * values[k], 6) for k in KS}}
            )
    with (OUT / "experiment3_main_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    # Paired source-group bootstrap.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in keys:
        groups[source_group[key]].append(key)
    group_keys = sorted(groups)
    group_index = {name: index for index, name in enumerate(group_keys)}
    key_group = np.array([group_index[source_group[key]] for key in keys])
    group_count = np.bincount(key_group, minlength=len(group_keys)).astype(np.float64)

    def bootstrap(observations: dict[tuple[str, str], float], k: int) -> dict[str, Any]:
        values = np.array([observations[key] for key in keys], dtype=np.float64)
        group_sum = np.bincount(key_group, weights=values, minlength=len(group_keys))
        point = float(values.mean())
        draws = rng.integers(0, len(group_keys), size=(BOOTSTRAP_REPLICATES, len(group_keys)))
        totals = group_sum[draws].sum(axis=1)
        counts = group_count[draws].sum(axis=1)
        samples = np.sort(totals / np.maximum(counts, 1))
        return {
            "delta_pp": round(100 * point, 4),
            "ci95_pp": [
                round(100 * float(samples[int(0.025 * len(samples))]), 4),
                round(100 * float(samples[int(0.975 * len(samples))]), 4),
            ],
        }

    def contrast(a_labels: list[str], b_labels: list[str], k: int) -> dict[str, Any]:
        """Average the per-query difference within a replication first, then across seeds."""
        observations = {
            key: sum(
                per_query[a][key][k] - per_query[b][key][k]
                for a, b in zip(a_labels, b_labels, strict=True)
            ) / len(a_labels)
            for key in keys
        }
        return bootstrap(observations, k)

    contrasts: dict[str, Any] = {}
    for tag, (arm_a, arm_b) in REPLICATIONS.items():
        contrasts[f"{tag}: armB - armA"] = {
            f"overall_R@{k}": contrast([arm_b], [arm_a], k) for k in KS
        }
    if len(REPLICATIONS) > 1:
        a_labels = [pair[0] for pair in REPLICATIONS.values()]
        b_labels = [pair[1] for pair in REPLICATIONS.values()]
        contrasts["seed-averaged: armB - armA"] = {
            f"overall_R@{k}": contrast(b_labels, a_labels, k) for k in KS
        }
        contrasts["seed-averaged: armB - parent"] = {
            f"overall_R@{k}": contrast(b_labels, [args.baseline] * len(b_labels), k)
            for k in KS
        }
        contrasts["seed-averaged: armB - QT-on-P"] = {
            f"overall_R@{k}": contrast(b_labels, ["QT-on-P"] * len(b_labels), k) for k in KS
        }

    for tag, (arm_a, arm_b) in CANDIDATE_REPLAYS.items():
        if arm_a not in per_query or arm_b not in per_query:
            continue
        contrasts[f"{tag}: armB - armA"] = {
            f"overall_R@{k}": contrast([arm_b], [arm_a], k) for k in KS
        }
        # Direct question: does the gain depend on the B13 candidate source?
        reference = REPLICATIONS["seed13"][1]
        if reference in per_query:
            contrasts[f"{tag}: armB({tag}) - armB(seed13)"] = {
                f"overall_R@{k}": contrast([arm_b], [reference], k) for k in KS
            }
            contrasts[f"{tag}: armA({tag}) - armA(seed13)"] = {
                f"overall_R@{k}": contrast([arm_a], [REPLICATIONS["seed13"][0]], k)
                for k in KS
            }

    spread = {}
    for k in KS:
        values = [
            round(
                100
                * sum(per_query[arm_b][key][k] - per_query[arm_a][key][k] for key in keys)
                / len(keys),
                4,
            )
            for arm_a, arm_b in REPLICATIONS.values()
        ]
        spread[f"armB - armA R@{k}"] = {
            "per_seed_pp": values,
            "mean_pp": round(sum(values) / len(values), 4),
            "range_pp": round(max(values) - min(values), 4),
        }

    report = {
        "support": "frozen FINAL_RERANK retained bags; identical P for every view",
        "queries": len(keys),
        "source_groups": len(group_keys),
        "replications": {tag: {"arm_a": pair[0], "arm_b": pair[1]}
                         for tag, pair in REPLICATIONS.items()},
        "candidate_replays": {tag: {"arm_a": pair[0], "arm_b": pair[1]}
                              for tag, pair in CANDIDATE_REPLAYS.items()},
        "main_table": table_rows,
        "contrasts_source_group_bootstrap": contrasts,
        "across_seed_spread": spread,
        "notes": [
            "QT-on-P is recomputed from frozen QT scores restricted to P.",
            "Bootstrap averages within query first, then resamples whole source groups.",
            "Dev is source-group disjoint from every Experiment-3 train query.",
            "Across seeds, the per-query difference is averaged first, then bootstrapped.",
        ],
    }
    (OUT / "EXPERIMENT3_COMPARISON.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in table_rows:
        print(json.dumps(row), flush=True)
    print(json.dumps(contrasts, indent=2), flush=True)
    print(json.dumps(spread, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

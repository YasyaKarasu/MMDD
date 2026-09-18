#!/usr/bin/env python
"""Aggregate WITNESS_DIAGNOSTIC.jsonl.gz into the Experiment-1 tables.

All rates here are explicit about their denominator.  Because the witness label
source is a strict subset of the qrels set, every "no witness" outcome is
`unknown`, never `verified_negative`; nothing in this file may be read as
"this target has no correct witness".
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "work/witness_diagnostic_20260916"
KS = (10, 20, 50)

# furthest_stage -> where the witness was actually lost
LOST_AT = {
    "absent": "not_in_pre_retention",
    "pre_retention": "lost_at_dedup",
    "dedup": "lost_at_top20",
    "top20": "lost_at_budget4",
    "retained": "retained",
}
CLASS_ORDER = [
    "retained",
    "lost_at_budget4",
    "lost_at_top20",
    "lost_at_dedup",
    "not_in_pre_retention",
]


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", default=str(IN / "WITNESS_DIAGNOSTIC.jsonl.gz"))
    args = parser.parse_args()
    diagnostic = Path(args.diagnostic)

    # ---- pass 1: per (endpoint, query, target) roll-up --------------------- #
    targets: dict[tuple[str, str, str], dict[str, Any]] = {}
    witness_rows: list[dict[str, Any]] = []
    competitor_rows: list[dict[str, Any]] = []
    for record in rows(diagnostic):
        key = (record["endpoint"], record["query_id"], record["target_id"])
        entry = targets.get(key)
        if entry is None:
            entry = targets[key] = {
                "endpoint": record["endpoint"],
                "family": record["family"],
                "seed": record["seed"],
                "query_id": record["query_id"],
                "query_kind": record["query_kind"],
                "target_id": record["target_id"],
                "target_is_positive": record["target_is_positive"],
                "target_source": record.get("target_source"),
                "cohorts": record.get("cohorts") or [],
                "ranks": record.get("ranks") or {},
                "verified_witness_ids": [],
                "best_stage": "absent",
                "best_exact_stage": "absent",
            }
        if record["witness_label"] == "verified_positive":
            witness_rows.append(record)
            entry["verified_witness_ids"].append(record["evidence_id"])
            if CLASS_ORDER.index(LOST_AT[record["furthest_stage"]]) < CLASS_ORDER.index(
                LOST_AT[entry["best_stage"]]
            ):
                entry["best_stage"] = record["furthest_stage"]
            if CLASS_ORDER.index(LOST_AT[record["exact_stage"]]) < CLASS_ORDER.index(
                LOST_AT[entry["best_exact_stage"]]
            ):
                entry["best_exact_stage"] = record["exact_stage"]
        else:
            competitor_rows.append(record)

    # ---- positive-target denominators ------------------------------------- #
    # The heavy pass only emits rows for targets that have a witness label or
    # are a Top-10 competitor.  Population sizes come from the frozen round.
    population = json.loads((IN / "POPULATION.json").read_text()) if (IN / "POPULATION.json").is_file() else {}

    per_endpoint = defaultdict(Counter)
    for entry in targets.values():
        if not entry["target_is_positive"]:
            continue
        bucket = per_endpoint[entry["endpoint"]]
        bucket["positive_targets"] += 1
        if not entry["verified_witness_ids"]:
            # Row exists only to carry the explicit `unknown` label.
            bucket["positive_targets_unknown"] += 1
            continue
        bucket["positive_targets_with_verified_witness"] += 1
        bucket[LOST_AT[entry["best_stage"]]] += 1

    # ---- main metric ------------------------------------------------------- #
    main_rows: list[dict[str, Any]] = []
    for endpoint in sorted(per_endpoint):
        bucket = per_endpoint[endpoint]
        labelled = bucket["positive_targets_with_verified_witness"]
        reached = sum(bucket[name] for name in CLASS_ORDER if name != "not_in_pre_retention")
        retained = bucket["retained"]
        main_rows.append(
            {
                "endpoint": endpoint,
                "positive_targets_in_c100": bucket["positive_targets"],
                "positive_targets_with_verified_witness": labelled,
                "positive_targets_unknown": bucket["positive_targets_unknown"],
                "unknown_share_pct": round(
                    100 * bucket["positive_targets_unknown"] / bucket["positive_targets"], 4
                ) if bucket["positive_targets"] else None,
                "witness_reached_pre_retention": reached,
                "witness_survived_to_budget4": retained,
                "retention_rate_given_reached_pct": round(100 * retained / reached, 4) if reached else None,
                "retention_rate_given_labelled_pct": round(100 * retained / labelled, 4) if labelled else None,
                "share_of_all_positives_pct": round(
                    100 * retained / bucket["positive_targets"], 4
                ) if bucket["positive_targets"] else None,
                "not_in_pre_retention": bucket["not_in_pre_retention"],
                "lost_at_dedup": bucket["lost_at_dedup"],
                "lost_at_top20": bucket["lost_at_top20"],
                "lost_at_budget4": bucket["lost_at_budget4"],
            }
        )
    write_csv(IN / "witness_retention_summary.csv", main_rows)

    # ---- loss-stage table over witness *assets* ---------------------------- #
    stage_rows: list[dict[str, Any]] = []
    stage_counter: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for record in witness_rows:
        stage_counter[(record["endpoint"], LOST_AT[record["furthest_stage"]])]["assets"] += 1
        stage_counter[(record["endpoint"], LOST_AT[record["furthest_stage"]])][
            f"modality_{record['modality']}"
        ] += 1
    for (endpoint, stage), counter in sorted(stage_counter.items()):
        stage_rows.append(
            {
                "endpoint": endpoint,
                "outcome": stage,
                "witness_assets": counter["assets"],
                "text": counter["modality_text"],
                "image": counter["modality_image"],
            }
        )
    write_csv(IN / "witness_stage_loss.csv", stage_rows)

    # ---- per-target classification ---------------------------------------- #
    target_rows: list[dict[str, Any]] = []
    for entry in sorted(targets.values(), key=lambda item: (item["endpoint"], item["query_id"], item["target_id"])):
        if not entry["target_is_positive"]:
            continue
        ranks = entry["ranks"]
        labelled = bool(entry["verified_witness_ids"])
        target_rows.append(
            {
                "endpoint": entry["endpoint"],
                "family": entry["family"],
                "query_id": entry["query_id"],
                "query_kind": entry["query_kind"],
                "target_id": entry["target_id"],
                "target_source": entry["target_source"],
                "cohorts": "|".join(entry["cohorts"]),
                "verified_witness_count": len(entry["verified_witness_ids"]),
                "outcome": LOST_AT[entry["best_stage"]] if labelled else "annotation_unknown",
                "outcome_exact_id_only": LOST_AT[entry["best_exact_stage"]] if labelled else "annotation_unknown",
                "qt_rank": ranks.get("QT-only"),
                "student_d1_rank": ranks.get("Student-D1-Path"),
                "student_lse_rank": ranks.get("Student-LSE-Path"),
                "teacher_lse_rank": ranks.get("Teacher-LSE-Path"),
                "qt_top10": int(ranks.get("QT-only") is not None and ranks["QT-only"] < 10),
                "teacher_top10": int(
                    ranks.get("Teacher-LSE-Path") is not None and ranks["Teacher-LSE-Path"] < 10
                ),
            }
        )
    write_csv(IN / "per_target_classification.csv", target_rows)

    # ---- cross-tab: witness outcome x scorer success ----------------------- #
    # Restricted to targets that carry a verified witness label, so the
    # `annotation_unknown` row is the only one with no witness evidence.
    scorers = {
        "QT-only": "qt_rank",
        "Student-D1-Path": "student_d1_rank",
        "Student-LSE-Path": "student_lse_rank",
        "Teacher-LSE-Path": "teacher_lse_rank",
    }
    crosstab: Counter = Counter()
    for row in target_rows:
        for scorer, field in scorers.items():
            rank = row[field]
            crosstab[(row["outcome"], scorer, "targets")] += 1
            if rank is not None and rank < 10:
                crosstab[(row["outcome"], scorer, "top10")] += 1
            if rank is not None and rank < 20:
                crosstab[(row["outcome"], scorer, "top20")] += 1
            if rank is not None and rank < 50:
                crosstab[(row["outcome"], scorer, "top50")] += 1
    write_csv(
        IN / "outcome_by_scorer.csv",
        [
            {
                "outcome": outcome,
                "scorer": scorer,
                "targets": crosstab[(outcome, scorer, "targets")],
                "top10": crosstab[(outcome, scorer, "top10")],
                "top20": crosstab[(outcome, scorer, "top20")],
                "top50": crosstab[(outcome, scorer, "top50")],
                "top10_pct": round(
                    100 * crosstab[(outcome, scorer, "top10")]
                    / max(1, crosstab[(outcome, scorer, "targets")]),
                    4,
                ),
                "top50_pct": round(
                    100 * crosstab[(outcome, scorer, "top50")]
                    / max(1, crosstab[(outcome, scorer, "targets")]),
                    4,
                ),
            }
            for outcome in CLASS_ORDER + ["annotation_unknown"]
            for scorer in scorers
        ],
    )

    # ---- fixed-207 cohort --------------------------------------------------- #
    cohort_rows: list[dict[str, Any]] = []
    by_cohort: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        for cohort in (row["cohorts"] or "").split("|"):
            if cohort:
                by_cohort[(row["endpoint"], cohort)].append(row)
    for (endpoint, cohort), items in sorted(by_cohort.items()):
        counts = Counter(item["outcome"] for item in items)
        cohort_rows.append(
            {
                "endpoint": endpoint,
                "cohort": cohort,
                "targets": len(items),
                **{name: counts[name] for name in CLASS_ORDER + ["annotation_unknown"]},
                "qt_top10": sum(item["qt_top10"] for item in items),
                "teacher_top10": sum(item["teacher_top10"] for item in items),
            }
        )
    write_csv(IN / "cohort_summary.csv", cohort_rows)

    # ---- failure decomposition --------------------------------------------- #
    decomposition: list[dict[str, Any]] = []
    for row in target_rows:
        if row["outcome"] == "annotation_unknown":
            continue
        decomposition.append(
            {
                "endpoint": row["endpoint"],
                "outcome": row["outcome"],
                "qt_top10": row["qt_top10"],
                "teacher_top10": row["teacher_top10"],
            }
        )
    write_csv(IN / "failure_decomposition.csv", decomposition)

    summary = {
        "positive_targets_by_endpoint": {
            endpoint: dict(bucket) for endpoint, bucket in sorted(per_endpoint.items())
        },
        "witness_assets": len(witness_rows),
        "competitor_evidence_rows": len(competitor_rows),
        "main_metric": main_rows,
    }
    text = json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    (IN / "SUMMARY.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

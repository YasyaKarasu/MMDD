#!/usr/bin/env python
"""Edge-level Teacher diagnostics on the verified witness edges.

The audit left one item Unknown: "QE/ET在verified edges上的能力" -- whether the
Teacher's individual QE and ET scores actually separate true bridging edges from
the edges the Student happened to retain.  This script measures exactly that,
plus the per-target witness contribution the Experiment-1 spec asks for
("对verified retained目标观察正确witness的QE、ET、sum、LSE贡献和与相同模态竞争者的margin").

Scores come from the frozen cache plus the separately written witness extension;
the frozen sqlite is never modified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "work/witness_diagnostic_20260916"
RERANK = ROOT / "work/final_rerank_20260916/FINAL_RERANK"


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_scores(path: Path) -> dict[tuple[str, str], float]:
    if not path.is_file():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    values = {
        (str(source), str(destination)): float(score)
        for source, destination, score in connection.execute(
            "SELECT source_id, destination_id, score FROM scores"
        )
    }
    connection.close()
    return values


def logsumexp(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values))


def auc(positive: list[float], negative: list[float]) -> float | None:
    """Rank-based AUC with average ranks for ties (Mann-Whitney)."""
    if not positive or not negative:
        return None
    combined = sorted([*((v, 1) for v in positive), *((v, 0) for v in negative)])
    ranks: dict[int, float] = {}
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        average = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1
    positive_rank_sum = sum(
        ranks[position] for position, (_, label) in enumerate(combined) if label == 1
    )
    n_pos, n_neg = len(positive), len(negative)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    frozen = load_scores(RERANK / "scores/teacher_pair_scores.sqlite")
    extension = load_scores(IN / "witness_teacher_pairs.sqlite")
    scores = {**frozen, **extension}
    print(json.dumps({"event": "scores", "frozen": len(frozen),
                      "extension": len(extension), "merged": len(scores)}), flush=True)

    membership = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "candidates/path_membership.jsonl.gz")
    }

    # ---- witness bookkeeping ---------------------------------------------- #
    witness_assets: dict[tuple[str, str], set[str]] = defaultdict(set)   # (q,t) -> {e}
    stages: dict[tuple[str, str, str], str] = {}
    positives: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in rows(IN / "WITNESS_DIAGNOSTIC.jsonl.gz"):
        key = (record["query_id"], record["target_id"])
        if record["target_is_positive"]:
            positives[record["endpoint"]].add(record["target_id"])
        if record["witness_label"] != "verified_positive":
            continue
        witness_assets[key].add(record["evidence_id"])
        stages[(record["query_id"], record["target_id"], record["evidence_id"])] = record["furthest_stage"]

    # ---- edge-level QE/ET separation --------------------------------------- #
    qe_positive: list[float] = []
    qe_negative: list[float] = []
    et_positive: list[float] = []
    et_negative: list[float] = []
    edge_rows: list[dict[str, Any]] = []
    seen_qe: set[tuple[str, str]] = set()
    seen_et: set[tuple[str, str]] = set()
    for (endpoint, query_id), record in membership.items():
        c100 = {str(value) for value in record["c100_ids"]}
        for target in record["targets"]:
            target_id = str(target["target_id"])
            if target_id not in c100:
                continue
            is_positive = target_id in positives[endpoint]
            for path in target.get("retained_paths", []):
                evidence_id = str(path["evidence_id"])
                modality = str(path.get("evidence_type") or "")
                qe = scores.get((query_id, evidence_id))
                et = scores.get((evidence_id, target_id))
                if qe is None or et is None:
                    continue
                is_witness = is_positive and evidence_id in witness_assets.get((query_id, target_id), set())
                row = {
                    "endpoint": endpoint,
                    "query_id": query_id,
                    "target_id": target_id,
                    "target_is_positive": is_positive,
                    "evidence_id": evidence_id,
                    "modality": modality,
                    "is_verified_witness": is_witness,
                    "witness_stage": stages.get((query_id, target_id, evidence_id)),
                    "teacher_query_evidence_score": qe,
                    "teacher_evidence_target_score": et,
                    "teacher_path_score": qe + et,
                }
                edge_rows.append(row)
                if is_witness:
                    if (query_id, evidence_id) not in seen_qe:
                        seen_qe.add((query_id, evidence_id))
                        qe_positive.append(qe)
                    if (evidence_id, target_id) not in seen_et:
                        seen_et.add((evidence_id, target_id))
                        et_positive.append(et)
                else:
                    qe_negative.append(qe)
                    et_negative.append(et)

    with (IN / "witness_edge_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)

    def summarize(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "mean": round(sum(ordered) / len(ordered), 5),
            "median": round(ordered[len(ordered) // 2], 5),
            "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 5),
        }

    edge_report = {
        "query_evidence": {
            "verified_witness_edges": summarize(qe_positive),
            "other_retained_edges": summarize(qe_negative),
            "auc_witness_vs_other": auc(qe_positive, qe_negative),
        },
        "evidence_target": {
            "verified_witness_edges": summarize(et_positive),
            "other_retained_edges": summarize(et_negative),
            "auc_witness_vs_other": auc(et_positive, et_negative),
        },
        "note": (
            "AUC 0.5 means the Teacher's single-edge scores do not separate true "
            "bridging edges from retained non-witness edges at all."
        ),
    }

    # ---- per-target witness contribution ----------------------------------- #
    contribution_rows: list[dict[str, Any]] = []
    for (endpoint, query_id), record in membership.items():
        c100 = [str(value) for value in record["c100_ids"]]
        positive = positives[endpoint]
        # best same-modality competitor edge score per modality, over C100
        competitor_best: dict[str, float] = {}
        for target in record["targets"]:
            target_id = str(target["target_id"])
            if target_id not in c100 or target_id in positive:
                continue
            for path in target.get("retained_paths", []):
                evidence_id = str(path["evidence_id"])
                qe = scores.get((query_id, evidence_id))
                et = scores.get((evidence_id, target_id))
                if qe is None or et is None:
                    continue
                modality = str(path.get("evidence_type") or "")
                value = qe + et
                competitor_best[modality] = max(competitor_best.get(modality, -math.inf), value)
        for target in record["targets"]:
            target_id = str(target["target_id"])
            if target_id not in c100 or target_id not in positive:
                continue
            witnesses = witness_assets.get((query_id, target_id), set())
            retained = target.get("retained_paths", [])
            if not retained:
                continue
            path_scores = []
            for path in retained:
                evidence_id = str(path["evidence_id"])
                qe = scores.get((query_id, evidence_id))
                et = scores.get((evidence_id, target_id))
                path_scores.append(None if qe is None or et is None else qe + et)
            valid = [value for value in path_scores if value is not None]
            if not valid:
                continue
            target_lse = logsumexp(valid)
            for position, path in enumerate(retained):
                evidence_id = str(path["evidence_id"])
                if evidence_id not in witnesses:
                    continue
                qe = scores.get((query_id, evidence_id))
                et = scores.get((evidence_id, target_id))
                if qe is None or et is None:
                    continue
                modality = str(path.get("evidence_type") or "")
                contribution_rows.append(
                    {
                        "endpoint": endpoint,
                        "query_id": query_id,
                        "target_id": target_id,
                        "evidence_id": evidence_id,
                        "modality": modality,
                        "teacher_query_evidence_score": qe,
                        "teacher_evidence_target_score": et,
                        "teacher_path_score": qe + et,
                        "retained_path_count": len(retained),
                        "witness_is_best_retained": int(
                            position == max(range(len(retained)),
                                            key=lambda index: (path_scores[index] is not None,
                                                               path_scores[index] or -math.inf))
                        ) if any(value is not None for value in path_scores) else 0,
                        "witness_lse_share": round(math.exp(qe + et - target_lse), 6),
                        "target_teacher_lse": target_lse,
                        "best_same_modality_competitor_path_score": (
                            None if modality not in competitor_best else competitor_best[modality]
                        ),
                        "margin_vs_best_same_modality_competitor": (
                            None if modality not in competitor_best
                            else round((qe + et) - competitor_best[modality], 6)
                        ),
                    }
                )
    with (IN / "witness_contribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contribution_rows[0]))
        writer.writeheader()
        writer.writerows(contribution_rows)

    by_modality = Counter()
    margins = defaultdict(list)
    for row in contribution_rows:
        by_modality[row["modality"]] += 1
        by_modality[f"{row['modality']}|best_retained"] += row["witness_is_best_retained"]
        if row["margin_vs_best_same_modality_competitor"] is not None:
            margins[row["modality"]].append(row["margin_vs_best_same_modality_competitor"])

    report = {
        "edge_quality": edge_report,
        "witness_contribution": {
            "rows": len(contribution_rows),
            "by_modality": dict(by_modality),
            "margin_vs_same_modality_competitor": {
                modality: {
                    "n": len(values),
                    "median": round(sorted(values)[len(values) // 2], 5),
                    "share_negative_pct": round(
                        100 * sum(1 for value in values if value < 0) / len(values), 3
                    ),
                }
                for modality, values in sorted(margins.items())
            },
        },
    }
    (IN / "WITNESS_SCORE_ANALYSIS.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

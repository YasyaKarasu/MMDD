"""Recount saved mechanism outputs and annotation coverage without model/config loading."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def annotation_coverage(repo: Path) -> dict:
    dataset = repo / (
        "output_mm_joinability_entitables_20000_retry100_rounds5_"
        "qwen35_final_survivor_context_gaussian_v9"
    )
    split_path = repo / "work/stage1_optimization_r10_20260907/taskA_protocol/splits.json"
    splits = json.loads(split_path.read_text())["query_ids"]
    by_query = {query: split for split, queries in splits.items() for query in queries}
    support = defaultdict(lambda: defaultdict(set))
    for record in read_jsonl(dataset / "evidence_recoveries/part-00000.jsonl"):
        pair = (record["query_table_id"], record["target_table_id"])
        support[by_query[pair[0]]][pair].add(record["query_row_id"])
    counts = {}
    for split, pairs in support.items():
        histogram = Counter(map(len, pairs.values()))
        ge3 = sum(count for rows, count in histogram.items() if rows >= 3)
        counts[split] = {
            "implicit_positive_pairs": len(pairs),
            "known_recovered_row_count_histogram": dict(sorted(histogram.items())),
            "known_rows_at_least_3_pairs": ge3,
            "known_rows_at_least_3_rate": ge3 / len(pairs),
        }
    query_rows = Counter(len(row["rows"]) for row in read_jsonl(dataset / "query_tables/part-00000.jsonl"))
    assert query_rows == {5: 14994}
    assert sum(value["implicit_positive_pairs"] for value in counts.values()) == 8359
    return {"r10_protocol_splits": counts, "query_row_histogram": dict(query_rows),
            "definition": "Distinct confirmed query rows per positive query-target pair; not a capability upper bound"}


def analyze(repo: Path) -> dict:
    root = repo / "work/stage1_optimization_r12_20260908/taskF_end_to_end"
    fa = read_jsonl(root / "f_a_predictions.jsonl")
    full = read_jsonl(root / "full_chain_predictions.jsonl")
    metrics = json.loads((root / "metrics.json").read_text())
    counts = Counter()
    strata = defaultdict(Counter)
    conditions = defaultdict(Counter)
    mismatches = []
    coverage = defaultdict(Counter)
    accepted_rows = []
    full_by_system = defaultdict(dict)
    for record in full:
        full_by_system[record["system"]][record["query_id"]] = record
        if record["system"] != "f1_union_direct":
            continue
        counts["queries"] += 1
        positive_ids = set(record["positive_target_ids"])
        counts["all_gold_pairs"] += len(positive_ids)
        candidates = [*record["stage2"]["reranked_candidates"],
                      *record["stage2"]["unattempted_candidates"]]
        for candidate in candidates:
            positive = candidate["target_id"] in positive_ids
            group = strata[record["query_kind"]]
            counts["candidate_pairs"] += 1
            counts["gold_pairs_in_queue"] += int(positive)
            group["candidate_pairs"] += 1
            group["gold_pairs_in_queue"] += int(positive)
            verification = candidate.get("verification") or {}
            accepted = bool(verification.get("joinable"))
            outcome = "qrel_positive" if positive else "qrel_unlisted"
            if accepted:
                counts[f"accepted_{outcome}"] += 1
                counts[f"accepted_branch_{candidate['final_branch']}"] += 1
                group[f"accepted_{outcome}"] += 1
                coverage[outcome][str(verification["coverage"])] += 1
                accepted_rows.append({
                    "query_kind": record["query_kind"], "qrel_positive": positive,
                    "branch": candidate["final_branch"],
                    "coverage": verification["coverage"],
                    "mean_similarity": verification["mean_similarity"],
                })
            evidence = candidate["branches"].get("evidence")
            if evidence is None:
                counts["candidate_without_evidence_branch"] += 1
                continue
            counts[f"evidence_status_{evidence['status']}"] += 1
            if evidence.get("verification"):
                counts["evidence_accepted"] += int(evidence["verification"]["joinable"])
            for row in evidence["rows"]:
                counts["full_row_outputs"] += 1
                counts["full_nonempty_outputs"] += bool(row["value"])
                counts["full_rows_with_localized_evidence"] += row.get("evidence") is not None
    for record in fa:
        for condition, result in record["conditions"].items():
            count = conditions[condition]
            count["pairs"] += 1
            count["recoverable_rows"] += result["recoverable_rows"]
            count["routed_supported_rows"] += len(result["routed_support_rows"])
            for row in result["rows"]:
                count["row_outputs"] += 1
                count["nonempty_outputs"] += bool(row["generated_value"])
                count["rows_with_chosen_evidence"] += row["evidence_id"] is not None
                count["chosen_evidence_supported_rows"] += bool(row["evidence_supported"])
                count["truth_available_rows"] += bool(row["truth_available"])
                count["correct_value_recoveries"] += bool(row["correct_value_recovery"])
            if condition in ("retrieved", "oracle_evidence"):
                routed = set(result["routed_support_rows"])
                chosen = {row["row_id"] for row in result["rows"] if row["evidence_supported"]}
                if routed != chosen:
                    mismatches.append({"condition": condition,
                        "routed_count": len(routed), "chosen_supported_count": len(chosen)})
    one = full_by_system["f1_union_direct"]
    other = full_by_system["frozen_fusion"]
    assert one.keys() == other.keys()
    identical = all(one[q]["stage2"] == other[q]["stage2"] for q in one)
    assert identical == metrics["inputs"]["queues_identical"]
    declared = metrics["by_system"]["f1_union_direct"]
    assert counts["accepted_qrel_positive"] == declared["true_final_joins"]
    assert counts["accepted_qrel_unlisted"] == declared["false_final_joins"]
    assert counts["queries"] == 128
    assert counts["candidate_pairs"] == sum(len(row["queue_details"]) for row in one.values())
    assert counts["accepted_qrel_positive"] / counts["gold_pairs_in_queue"] == declared["final_join_recall_given_positive_in_queue"]
    for condition, count in conditions.items():
        assert count["recoverable_rows"] == metrics["f_a"]["by_condition"][condition]["recoverable_rows"]
    return {
        "sources": [str(p.relative_to(repo)) for p in [root / "f_a_predictions.jsonl",
                    root / "full_chain_predictions.jsonl", root / "metrics.json"]],
        "scope": "R12 frozen 128 dev queries; count identical systems once",
        "counts": dict(counts), "strata": dict(strata),
        "fa_conditions": dict(conditions),
        "routed_vs_chosen_mismatches": mismatches,
        "accepted_coverage_distribution": dict(coverage),
        "accepted_candidate_rows": accepted_rows,
        "systems_have_identical_predictions": identical,
        "reported_value_denominator": declared["correct_value_denominator"],
        "precision_against_existing_qrels": counts["accepted_qrel_positive"] /
            (counts["accepted_qrel_positive"] + counts["accepted_qrel_unlisted"]),
        "caveat": "qrel_unlisted is the evaluator's false-positive definition; independent review is pending",
        "checks": "Saved join counts, condition denominators and duplicate-system identity reconciled",
        "annotation_coverage": annotation_coverage(repo),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "accepted_candidate_rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

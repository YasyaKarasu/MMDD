"""Paired B0/B1 bridge diagnostics and source-group bootstrap."""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_stage1_bridge import OUT, ROOT, write_json


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recall(row: dict[str, Any], ranking: list[str], k: int | None = None) -> float:
    positives = set(row["positive_target_ids"])
    values = ranking if k is None else ranking[:k]
    return len(positives & set(values)) / len(positives) if positives else 0.0


def own_metrics(row: dict[str, Any]) -> dict[str, float]:
    rankings = row["rankings"]
    return {
        "Direct_exact_R10": recall(row, row["D100_EXACT"], 10),
        "E_R10": recall(row, rankings["E_ONLY"], 10),
        "U_RawRecall": recall(row, row["U"]),
        "strict_EO_admitted": len(set(row.get("EO_EXACT", ()))) / len(row["positive_target_ids"]),
    }


def c100_metrics(own: dict[str, Any], teacher: dict[str, Any]) -> dict[str, float]:
    positives = set(own["positive_target_ids"])
    direct = set(own["D100_EXACT"][:100])
    union = set(own["U"])
    strict_eligible = positives & (union - direct)
    c100 = teacher["rankings"]["BT100_NO_T0"]
    admitted = strict_eligible & set(c100)
    return {
        "C100_Recall": recall(own, c100),
        "C100_T0_R10": recall(own, teacher["rankings"]["BT100_T0"], 10),
        "strict_EO_C100_admitted": len(admitted) / len(positives) if positives else 0.0,
    }


def paired(left: dict[str, float], right: dict[str, float], groups: dict[str, str], *, seed: int = 260914, replicates: int = 10000) -> dict[str, Any]:
    values = sorted(set(left) & set(right))
    if not values:
        raise ValueError("no paired query IDs")
    deltas = [right[q] - left[q] for q in values]
    win = sum(delta > 0 for delta in deltas)
    loss = sum(delta < 0 for delta in deltas)
    ties = len(deltas) - win - loss
    source_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, query_id in enumerate(values):
        source_to_indices[groups[query_id]].append(index)
    source_names = sorted(source_to_indices)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(replicates):
        sampled = [source_names[rng.randrange(len(source_names))] for _ in source_names]
        selected = [index for source in sampled for index in source_to_indices[source]]
        bootstrap.append(sum(deltas[index] for index in selected) / len(selected))
    bootstrap.sort()
    return {
        "queries": len(values),
        "sources": len(source_names),
        "mean_delta": sum(deltas) / len(deltas),
        "win": win,
        "loss": loss,
        "tie": ties,
        "win_rate_excluding_ties": win / (win + loss) if win + loss else None,
        "source_group_bootstrap": {
            "seed": seed,
            "replicates": replicates,
            "ci95_low": bootstrap[int(replicates * 0.025)],
            "ci95_high": bootstrap[int(replicates * 0.975) - 1],
        },
    }


def analyze(candidate: str = "B1", *, output_name: str | None = None) -> dict[str, Any]:
    b0_own_path = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation/rankings/H-C2-step000178/rankings.jsonl.gz"
    b0_t0_path = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation/teacher/H-C2-step000178/rankings.jsonl.gz"
    b1_own_path = OUT / f"evaluation/rankings/{candidate}/rankings.jsonl.gz"
    b1_t0_path = OUT / f"evaluation/teacher/{candidate}/rankings.jsonl.gz"
    b0_own = {row["query_id"]: row for row in read_rows(b0_own_path)}
    b0_t0 = {row["query_id"]: row for row in read_rows(b0_t0_path)}
    b1_own = {row["query_id"]: row for row in read_rows(b1_own_path)}
    b1_t0 = {row["query_id"]: row for row in read_rows(b1_t0_path)}
    common = sorted(set(b0_own) & set(b1_own) & set(b0_t0) & set(b1_t0))
    groups = {query_id: b0_own[query_id]["source_table_id"] for query_id in common}
    own_values = {stage: {metric: {} for metric in ("Direct_exact_R10", "E_R10", "U_RawRecall", "strict_EO_admitted")} for stage in ("B0", "B1")}
    c100_values = {stage: {metric: {} for metric in ("C100_Recall", "C100_T0_R10", "strict_EO_C100_admitted")} for stage in ("B0", "B1")}
    for query_id in common:
        for stage, own, teacher in (("B0", b0_own[query_id], b0_t0[query_id]), ("B1", b1_own[query_id], b1_t0[query_id])):
            for metric, value in own_metrics(own).items():
                own_values[stage][metric][query_id] = value
            for metric, value in c100_metrics(own, teacher).items():
                c100_values[stage][metric][query_id] = value
    paired_results = {}
    for metric in own_values["B0"]:
        paired_results[metric] = paired(own_values["B0"][metric], own_values["B1"][metric], groups)
    for metric in c100_values["B0"]:
        paired_results[metric] = paired(c100_values["B0"][metric], c100_values["B1"][metric], groups)
    absolute = {}
    for stage in ("B0", "B1"):
        metrics = {**own_values[stage], **c100_values[stage]}
        absolute[stage] = {
            metric: sum(values.values()) / len(values)
            for metric, values in metrics.items()
        }
    result = {
        "status": "B1_complete_B0_B1_paired",
        "scope": f"{candidate}; modern initialization only; historical C1/C2 supervision retained",
        "absolute_query_macro": absolute,
        "paired_B1_minus_B0": paired_results,
        "first_cliff_decision": {
            "status": "candidate_cliff_requires_confirmation",
            "reason": "B1 shows aligned Direct/U and C100/T0 movement; paired bootstrap is reported before extending the ladder",
            "next_action": "repeat B1 on seed29 or a pre-registered second initialization control before treating init as causal cliff",
        },
        "artifacts": {"B0_own": str(b0_own_path), "B0_T0": str(b0_t0_path), "candidate_own": str(b1_own_path), "candidate_T0": str(b1_t0_path)},
    }
    suffix = "" if output_name is None else f"_{output_name}"
    write_json(OUT / f"BRIDGE_RESULTS{suffix}.json", result)
    write_json(OUT / f"PAIRED_STATISTICS{suffix}.json", paired_results)
    return result


def analyze_adjacent(candidate: str, baseline: str, *, output_name: str | None = None) -> dict[str, Any]:
    """Compare two adjacent bridge nodes on the same frozen query population.

    This is intentionally separate from ``analyze``: B0 is the historical
    replay baseline, whereas later ladder steps must be interpreted as the
    pre-registered one-factor change from their immediately preceding node.
    """
    baseline_own_path = OUT / f"evaluation/rankings/{baseline}/rankings.jsonl.gz"
    baseline_t0_path = OUT / f"evaluation/teacher/{baseline}/rankings.jsonl.gz"
    candidate_own_path = OUT / f"evaluation/rankings/{candidate}/rankings.jsonl.gz"
    candidate_t0_path = OUT / f"evaluation/teacher/{candidate}/rankings.jsonl.gz"
    baseline_own = {row["query_id"]: row for row in read_rows(baseline_own_path)}
    baseline_t0 = {row["query_id"]: row for row in read_rows(baseline_t0_path)}
    candidate_own = {row["query_id"]: row for row in read_rows(candidate_own_path)}
    candidate_t0 = {row["query_id"]: row for row in read_rows(candidate_t0_path)}
    common = sorted(set(baseline_own) & set(baseline_t0) & set(candidate_own) & set(candidate_t0))
    if not common:
        raise ValueError(f"no paired queries for {baseline}->{candidate}")
    groups = {query_id: baseline_own[query_id]["source_table_id"] for query_id in common}
    metric_names = ("Direct_exact_R10", "E_R10", "U_RawRecall", "strict_EO_admitted",
                    "C100_Recall", "C100_T0_R10", "strict_EO_C100_admitted")
    per_stage: dict[str, dict[str, dict[str, float]]] = {}
    for label, own, teacher in ((baseline, baseline_own, baseline_t0), (candidate, candidate_own, candidate_t0)):
        values = {metric: {} for metric in metric_names}
        for query_id in common:
            merged = {**own_metrics(own[query_id]), **c100_metrics(own[query_id], teacher[query_id])}
            for metric in metric_names:
                values[metric][query_id] = merged[metric]
        per_stage[label] = values
    paired_results = {metric: paired(per_stage[baseline][metric], per_stage[candidate][metric], groups)
                      for metric in metric_names}
    absolute = {label: {metric: sum(values.values()) / len(values)
                        for metric, values in per_stage[label].items()}
                for label in (baseline, candidate)}
    result = {
        "status": "adjacent_pair_complete",
        "scope": f"{baseline}->{candidate}; same frozen query population and T0 protocol",
        "queries": len(common),
        "absolute_query_macro": absolute,
        f"paired_{candidate}_minus_{baseline}": paired_results,
        "artifacts": {"baseline_own": str(baseline_own_path), "baseline_T0": str(baseline_t0_path),
                      "candidate_own": str(candidate_own_path), "candidate_T0": str(candidate_t0_path)},
    }
    suffix = "" if output_name is None else f"_{output_name}"
    write_json(OUT / f"BRIDGE_RESULTS{suffix}.json", result)
    write_json(OUT / f"PAIRED_STATISTICS{suffix}.json", paired_results)
    return result


def confirm_b1() -> dict[str, Any]:
    """Combine seed13/seed29 per-query deltas for the pre-registered cliff check."""
    b0_own_path = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation/rankings/H-C2-step000178/rankings.jsonl.gz"
    b0_t0_path = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation/teacher/H-C2-step000178/rankings.jsonl.gz"
    b0_own = {row["query_id"]: row for row in read_rows(b0_own_path)}
    b0_t0 = {row["query_id"]: row for row in read_rows(b0_t0_path)}
    candidates = {}
    for label in ("B1", "B1_seed29"):
        own_path = OUT / f"evaluation/rankings/{label}/rankings.jsonl.gz"
        t0_path = OUT / f"evaluation/teacher/{label}/rankings.jsonl.gz"
        candidates[label] = ({row["query_id"]: row for row in read_rows(own_path)},
                             {row["query_id"]: row for row in read_rows(t0_path)})
    common = sorted(set(b0_own) & set(b0_t0) & set(candidates["B1"][0]) & set(candidates["B1_seed29"][0]) &
                    set(candidates["B1"][1]) & set(candidates["B1_seed29"][1]))
    groups = {query_id: b0_own[query_id]["source_table_id"] for query_id in common}
    metric_names = ("Direct_exact_R10", "E_R10", "U_RawRecall", "strict_EO_admitted",
                    "C100_Recall", "C100_T0_R10", "strict_EO_C100_admitted")
    per_seed = {}
    for label, (own, teacher) in candidates.items():
        values = {}
        for query_id in common:
            own_values = own_metrics(own[query_id])
            c_values = c100_metrics(own[query_id], teacher[query_id])
            values[query_id] = {**own_values, **c_values}
        per_seed[label] = {metric: sum(row[metric] for row in values.values()) / len(values) for metric in metric_names}
    averaged = {metric: {} for metric in metric_names}
    for query_id in common:
        b0_values = {**own_metrics(b0_own[query_id]), **c100_metrics(b0_own[query_id], b0_t0[query_id])}
        for metric in metric_names:
            averaged[metric][query_id] = (sum(
                ({**own_metrics(candidates[label][0][query_id]), **c100_metrics(candidates[label][0][query_id], candidates[label][1][query_id])})[metric]
                for label in ("B1", "B1_seed29")
            ) / 2)
    paired_results = {metric: paired(
        {query_id: ({**own_metrics(b0_own[query_id]), **c100_metrics(b0_own[query_id], b0_t0[query_id])})[metric] for query_id in common},
        averaged[metric], groups)
        for metric in metric_names}
    strict_excludes_zero = paired_results["strict_EO_admitted"]["source_group_bootstrap"]["ci95_high"] < 0
    direct_u_excludes_zero = all(paired_results[name]["source_group_bootstrap"]["ci95_high"] < 0
                                 or paired_results[name]["source_group_bootstrap"]["ci95_low"] > 0
                                 for name in ("Direct_exact_R10", "U_RawRecall"))
    result = {
        "status": "confirmed_candidate_cliff" if strict_excludes_zero and direct_u_excludes_zero else "initialization_cliff_not_confirmed",
        "scope": "B1 modern initialization; two independent seeds; historical supervision and order held fixed",
        "queries": len(common),
        "per_seed_absolute_query_macro": per_seed,
        "paired_seed_averaged_B1_minus_B0": paired_results,
        "interpretation": {
            "strict_EO_ci_excludes_zero": strict_excludes_zero,
            "direct_and_u_both_ci_exclude_zero": direct_u_excludes_zero,
            "next_action": "proceed to B2 after preserving B1 as a non-causal initialization effect" if not (strict_excludes_zero and direct_u_excludes_zero) else "pause and confirm initialization with a pre-registered control",
        },
        "artifacts": {"B0_own": str(b0_own_path), "B0_T0": str(b0_t0_path),
                      "B1": str((OUT / "evaluation/rankings/B1/rankings.jsonl.gz")),
                      "B1_seed29": str((OUT / "evaluation/rankings/B1_seed29/rankings.jsonl.gz"))},
    }
    write_json(OUT / "B1_CONFIRMATION.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=(
        "B1", "B1_seed29", "B2", "B2_seed29", "B3", "B3_seed29",
        "B4", "B4_seed29", "B5_356", "B5_356_seed29",
        "B5_659", "B5_659_seed29", "B5", "B5_seed29",
        "B6", "B6_seed29", "B7", "B7_seed29"), default="B1")
    parser.add_argument("--baseline", choices=(
        "B0", "B1", "B1_seed29", "B2", "B2_seed29", "B3", "B3_seed29",
        "B4", "B4_seed29", "B5_356", "B5_356_seed29",
        "B5_659", "B5_659_seed29", "B5", "B5_seed29",
        "B6", "B6_seed29"))
    parser.add_argument("--output-name")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.confirm:
        result = confirm_b1()
    elif args.baseline:
        result = analyze_adjacent(args.candidate, args.baseline, output_name=args.output_name)
    else:
        result = analyze(args.candidate, output_name=args.output_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))

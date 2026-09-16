"""Apply the preregistered R30 C1 recipe-selection rule."""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from prepare_stage1_r27 import ROOT
from run_stage1_bridge import sha256, write_json


OUT = ROOT / "work/stage1_r30_c1_et_20260916"
BOOTSTRAP_SEED = 260914
BOOTSTRAP_REPLICATES = 10_000


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def target_recall(ranking: list[str], positives: list[str], cutoff: int = 10) -> float:
    truth = set(positives)
    return len(truth & set(ranking[:cutoff])) / len(truth)


def load_metric(generator: str) -> dict[str, dict[str, Any]]:
    path = OUT / f"teacher/{generator}/rankings.jsonl.gz"
    result = {}
    for row in rows(path):
        result[str(row["query_id"])] = {
            "value": target_recall(row["rankings"]["BT100_T0"], row["positive_target_ids"]),
            "kind": str(row["query_kind"]),
            "source_table_id": str(row["source_table_id"]),
        }
    return result


def source_cluster_bootstrap(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in query_rows:
        clusters[row["source_table_id"]].append(float(row["delta"]))
    source_ids = sorted(clusters)
    sums = np.asarray([sum(clusters[source_id]) for source_id in source_ids], dtype=np.float64)
    counts = np.asarray([len(clusters[source_id]) for source_id in source_ids], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 100):
        size = min(100, BOOTSTRAP_REPLICATES - start)
        sampled = rng.integers(0, len(source_ids), size=(size, len(source_ids)))
        estimates[start:start + size] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "queries": len(query_rows),
        "source_groups": len(source_ids),
        "estimate": sum(row["delta"] for row in query_rows) / len(query_rows),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": BOOTSTRAP_SEED,
        "estimator": "sample source groups, then sum selected query differences / selected query count",
    }


def select() -> dict[str, Any]:
    seed_results = []
    by_seed = {}
    for seed in (13, 29):
        baseline_name = f"F-P659_s{seed}"
        treatment_name = f"F-P-ETNAT659_s{seed}"
        baseline = load_metric(baseline_name)
        treatment = load_metric(treatment_name)
        if baseline.keys() != treatment.keys():
            raise ValueError(f"query population mismatch for seed {seed}")
        query_rows = []
        for query_id in sorted(baseline):
            before, after = baseline[query_id], treatment[query_id]
            if before["kind"] != after["kind"] or before["source_table_id"] != after["source_table_id"]:
                raise ValueError(f"query metadata mismatch: {query_id}")
            query_rows.append({
                "seed": seed,
                "query_id": query_id,
                "query_kind": before["kind"],
                "source_table_id": before["source_table_id"],
                "F-P": before["value"],
                "F-P-ETNAT": after["value"],
                "delta": after["value"] - before["value"],
            })
        by_seed[seed] = {row["query_id"]: row for row in query_rows}
        overall = sum(row["delta"] for row in query_rows) / len(query_rows)
        implicit = [row for row in query_rows if row["query_kind"] == "implicit"]
        implicit_delta = sum(row["delta"] for row in implicit) / len(implicit)
        seed_results.append({
            "seed": seed,
            "queries": len(query_rows),
            "C100_T0_R10_delta": overall,
            "implicit_C100_T0_R10_delta": implicit_delta,
            "wins": sum(row["delta"] > 0 for row in query_rows),
            "losses": sum(row["delta"] < 0 for row in query_rows),
            "ties": sum(row["delta"] == 0 for row in query_rows),
            "baseline_receipt_sha256": sha256(OUT / f"teacher/{baseline_name}/TEACHER_RECEIPT.json"),
            "treatment_receipt_sha256": sha256(OUT / f"teacher/{treatment_name}/TEACHER_RECEIPT.json"),
        })
    paired = []
    for query_id in sorted(by_seed[13]):
        if query_id not in by_seed[29]:
            raise ValueError(f"seed29 missing query {query_id}")
        a, b = by_seed[13][query_id], by_seed[29][query_id]
        if a["source_table_id"] != b["source_table_id"]:
            raise ValueError(f"seed source mismatch: {query_id}")
        paired.append({
            "query_id": query_id,
            "query_kind": a["query_kind"],
            "source_table_id": a["source_table_id"],
            "delta": (a["delta"] + b["delta"]) / 2,
            "seed13_delta": a["delta"],
            "seed29_delta": b["delta"],
        })
    bootstrap = source_cluster_bootstrap(paired)
    conditions = {
        "both_seed_C100_T0_point_estimates_nonnegative": all(row["C100_T0_R10_delta"] >= 0 for row in seed_results),
        "pooled_delta_at_least_0_5pp": bootstrap["estimate"] >= 0.005,
        "pooled_95CI_lower_above_zero": bootstrap["ci95"][0] > 0,
        "implicit_not_down_more_than_1pp_each_seed": all(row["implicit_C100_T0_R10_delta"] >= -0.01 for row in seed_results),
    }
    etnat_selected = all(conditions.values())
    selection = {
        "status": "selected",
        "recipe": "F-P-ETNAT" if etnat_selected else "F-P",
        "ETNAT_extra_benefit_gate": "pass" if etnat_selected else "fail",
        "conditions": conditions,
        "seed_results": seed_results,
        "pooled_source_cluster_bootstrap": bootstrap,
        "selection_policy": "F-P unless ETNAT clears every preregistered incremental-benefit condition",
        "exploratory_dev_selection": True,
    }
    statistics = OUT / "statistics"
    statistics.mkdir(parents=True, exist_ok=True)
    write_json(statistics / "ETNAT_VS_FP_BOOTSTRAP.json", {
        "selection": selection,
        "paired_queries": paired,
    })
    write_json(OUT / "C1_SELECTION.json", selection)
    return selection


if __name__ == "__main__":
    print(json.dumps(select(), ensure_ascii=False, indent=2))

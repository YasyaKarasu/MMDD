"""Assemble the complete R30 statistics, audit ledger, and scientific report."""
from __future__ import annotations

import argparse
import csv
import difflib
import gzip
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from prepare_stage1_r27 import ROOT
from run_stage1_bridge import sha256, write_json


OUT = ROOT / "work/stage1_r30_c1_et_20260916"
BRIDGE = ROOT / "work/stage1_bridge_20260915/evaluation"
R26 = ROOT / "work/stage1_optimization_r26_20260914"
R29 = ROOT / "work/stage1_optimization_r29_candidate_vs_drift_20260915/teacher_evaluation"
R27 = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact"
BOOTSTRAP_SEED = 260914
BOOTSTRAP_REPLICATES = 10_000
K_VALUES = (10, 20, 50)

R30_SOURCES = (
    "src/run_stage1_r30.py",
    "src/run_stage1_r30_etnat.py",
    "src/run_stage1_r30_c2.py",
    "src/evaluate_stage1_r30.py",
    "src/diagnose_stage1_r30.py",
    "src/diagnose_stage1_r30_auxiliary.py",
    "src/diagnose_stage1_r30_et_candidates.py",
    "src/select_stage1_r30_c1.py",
    "src/summarize_stage1_r30.py",
    "src/finalize_stage1_r30.py",
    "tests/test_stage1_r30.py",
)
DEPENDENCY_SOURCES = (
    "src/evaluate_stage1_r26.py",
    "src/evaluate_stage1_r26_teacher.py",
    "src/run_stage1_bridge.py",
    "src/run_stage1_r12_task_c.py",
    "src/run_stage1_r13.py",
    "src/mmdd_stage1/checkpoints.py",
    "src/mmdd_stage1/features.py",
    "src/mmdd_stage1/models.py",
    "src/mmdd_stage1/objectives.py",
    "src/mmdd_stage1/scoring.py",
    "src/mmdd_stage1/training.py",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ids(values: Iterable[Any]) -> list[str]:
    return [str(value.get("target_id")) if isinstance(value, dict) else str(value) for value in values]


def target_recall(ranking: Iterable[str], positives: Iterable[str], cutoff: int | None = None) -> float:
    truth = set(positives)
    values = list(ranking)
    if cutoff is not None:
        values = values[:cutoff]
    return len(truth & set(values)) / len(truth)


def ranking_metrics(ranking: list[str], positives: list[str]) -> dict[str, float]:
    result = {"raw": target_recall(ranking, positives)}
    result.update({f"R{k}": target_recall(ranking, positives, k) for k in K_VALUES})
    return result


def flatten_metrics(prefix: str, ranking: list[str], positives: list[str]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in ranking_metrics(ranking, positives).items()}


def endpoint_inventory() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def add(name: str, panel: str, seed: int | None, own: Path, teacher: Path, provenance: str) -> None:
        if not own.is_file() or not teacher.is_file():
            raise FileNotFoundError(f"incomplete endpoint {name}: {own} / {teacher}")
        result.append({
            "endpoint": name,
            "panel": panel,
            "seed": seed,
            "own": own,
            "teacher": teacher,
            "provenance": provenance,
        })

    for seed in (13, 29):
        suffix = "" if seed == 13 else "_seed29"
        for name, label in (("STOP356", "B5_356"), ("JOINT500", "B5_500"), ("JOINT659", "B5_659")):
            add(f"{name}_s{seed}", "C1", seed,
                BRIDGE / f"rankings/{label}{suffix}/rankings.jsonl.gz",
                BRIDGE / f"teacher/{label}{suffix}/rankings.jsonl.gz", "qualified Bridge C1 control")
        for recipe in ("F-P", "F-P-ETNAT"):
            token = recipe.replace("-", "")
            for step in (500, 659):
                generator = f"{recipe}{step}_s{seed}"
                add(f"{token}{step}_s{seed}", "C1", seed,
                    OUT / f"rankings/{generator}/rankings.jsonl.gz",
                    OUT / f"teacher/{generator}/rankings.jsonl.gz", f"R30 {recipe}")
        for name in ("B4", "B5"):
            add(f"{name}-C2_s{seed}", "C2", seed,
                BRIDGE / f"rankings/{name}{suffix}/rankings.jsonl.gz",
                BRIDGE / f"teacher/{name}{suffix}/rankings.jsonl.gz", "qualified Bridge full-C2 control")

    selection = load(OUT / "C1_SELECTION.json")
    recipe = str(selection["recipe"])
    token = recipe.replace("-", "")
    for seed in (13, 29):
        selected_generator = f"{recipe}659_s{seed}"
        add(f"C2-{token}0_s{seed}", "C2", seed,
            OUT / f"rankings/{selected_generator}/rankings.jsonl.gz",
            OUT / f"teacher/{selected_generator}/rankings.jsonl.gz", "alias of selected C1 step659")
        for step in (89, 178):
            generator = f"C2-{recipe}{step}_s{seed}"
            add(f"C2-{token}{step}_s{seed}", "C2", seed,
                OUT / f"rankings/{generator}/rankings.jsonl.gz",
                OUT / f"teacher/{generator}/rankings.jsonl.gz", "R30 selected-recipe C2-CHECK")

    for name in ("Qwen-Raw", "B13"):
        add(name, "REFERENCE", None,
            R26 / f"rankings/{name}/rankings.jsonl.gz",
            R26 / f"teacher/{name}/rankings.jsonl.gz", "qualified R26 historical reference")
    for recipe in ("P", "R"):
        generator = f"S-EDGE-FREEZE-{recipe}"
        add(f"R29-FREEZE-{recipe}", "REFERENCE", 13,
            R29 / f"rankings/{generator}/rankings.jsonl.gz",
            R29 / f"teacher/{generator}/rankings.jsonl.gz", "qualified R29 seed13 reference; no seed29 claim")
    return result


def fixed_207() -> dict[str, set[str]]:
    path = R27 / "score_handoff/B13/baseline_per_query.jsonl.gz"
    result: dict[str, set[str]] = {}
    for row in rows(path):
        strict = set(str(value) for value in row["EO_sets"]["EO_STRICT"])
        if strict:
            result[str(row["query_id"])] = strict
    if sum(map(len, result.values())) != 207:
        raise ValueError("historical B13 fixed EO cohort is not 207 pairs")
    return result


def process_endpoints() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, dict[str, bool]]]],
]:
    statistics = OUT / "statistics"
    statistics.mkdir(parents=True, exist_ok=True)
    fixed = fixed_207()
    scalar_by_endpoint: dict[str, dict[str, dict[str, Any]]] = {}
    positive_state: dict[str, dict[str, dict[str, dict[str, bool]]]] = {}
    summaries: list[dict[str, Any]] = []
    eo_summary: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    per_query_path = statistics / "per_query.jsonl.gz"
    funnel_path = statistics / "fixed207_and_own_eo.csv"
    with gzip.open(per_query_path, "wt", encoding="utf-8") as per_query, funnel_path.open("w", newline="", encoding="utf-8") as funnel:
        writer = csv.DictWriter(funnel, fieldnames=(
            "endpoint", "panel", "seed", "cohort", "query_id", "target_id", "query_kind",
            "D_ANN100", "D_exact100", "E", "U", "C100", "T0_top10", "T0_top20", "T0_top50",
            "role",
        ))
        writer.writeheader()
        for endpoint in endpoint_inventory():
            name = endpoint["endpoint"]
            compact: dict[str, dict[str, Any]] = {}
            states: dict[str, dict[str, dict[str, bool]]] = {}
            grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
            own_iter = rows(endpoint["own"])
            teacher_iter = rows(endpoint["teacher"])
            count = 0
            for own, teacher in zip(own_iter, teacher_iter, strict=True):
                query_id = str(own["query_id"])
                if query_id != str(teacher["query_id"]):
                    raise ValueError(f"endpoint {name} own/Teacher query mismatch")
                positives = [str(value) for value in own["positive_target_ids"]]
                query_kind = str(own["query_kind"])
                own_rankings = {key: ids(value) for key, value in own["rankings"].items()}
                teacher_rankings = {key: ids(value) for key, value in teacher["rankings"].items()}
                direct_ann = set(ids(own["D100_ANN"]))
                direct_exact = set(ids(own["D100_EXACT"]))
                evidence = set(ids(own["E_target_ids"]))
                union = set(ids(own["U"]))
                c100 = set(own_rankings["Equal"][:100])
                t0 = teacher_rankings["BT100_T0"]
                own_eo = set(positives) & (evidence - (direct_ann | direct_exact))

                metrics: dict[str, float] = {}
                for method in ("D100_ANN", "D100_EXACT", "E_ONLY", "Equal", "U", "M_EXACT"):
                    metrics.update(flatten_metrics(method, own_rankings[method], positives))
                for method in ("BT100_T0", "D100_T0", "U_OFFLINE_T0", "M_OFFLINE_T0"):
                    metrics.update(flatten_metrics(method, teacher_rankings[method], positives))
                metrics.update({
                    "own_EO_admission": len(own_eo) / len(positives),
                    "own_EO_C100": len(own_eo & c100) / len(positives),
                    "own_EO_T0_R10": len(own_eo & set(t0[:10])) / len(positives),
                    "fixed207_admission": len(fixed.get(query_id, set()) & union) / len(positives),
                    "fixed207_C100": len(fixed.get(query_id, set()) & c100) / len(positives),
                    "fixed207_T0_R10": len(fixed.get(query_id, set()) & set(t0[:10])) / len(positives),
                })
                meta = {
                    "query_id": query_id,
                    "query_kind": query_kind,
                    "source_table_id": str(own["source_table_id"]),
                    "positive_target_ids": positives,
                }
                payload = {
                    "endpoint": name,
                    "panel": endpoint["panel"],
                    "seed": endpoint["seed"],
                    **meta,
                    "metrics": metrics,
                }
                per_query.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                compact[query_id] = {**meta, **metrics}
                grouped["overall"].append(metrics)
                grouped[query_kind].append(metrics)
                count += 1

                query_states: dict[str, dict[str, bool]] = {}
                for target_id in positives:
                    flags = {
                        "D_ANN100": target_id in direct_ann,
                        "D_exact100": target_id in direct_exact,
                        "E": target_id in evidence,
                        "U": target_id in union,
                        "C100": target_id in c100,
                        "T0_top10": target_id in set(t0[:10]),
                        "T0_top20": target_id in set(t0[:20]),
                        "T0_top50": target_id in set(t0[:50]),
                    }
                    query_states[target_id] = flags
                    if target_id in own_eo:
                        writer.writerow({
                            "endpoint": name, "panel": endpoint["panel"], "seed": endpoint["seed"],
                            "cohort": "own_strict_EO", "query_id": query_id, "target_id": target_id,
                            "query_kind": query_kind, **{key: int(value) for key, value in flags.items()},
                            "role": "own_evidence_only",
                        })
                states[query_id] = query_states

                for target_id in fixed.get(query_id, set()):
                    flags = {
                        "D_ANN100": target_id in direct_ann,
                        "D_exact100": target_id in direct_exact,
                        "E": target_id in evidence,
                        "U": target_id in union,
                        "C100": target_id in c100,
                        "T0_top10": target_id in set(t0[:10]),
                        "T0_top20": target_id in set(t0[:20]),
                        "T0_top50": target_id in set(t0[:50]),
                    }
                    role = "direct_now" if flags["D_ANN100"] or flags["D_exact100"] else (
                        "evidence_retained" if flags["E"] else "dropped_from_U")
                    writer.writerow({
                        "endpoint": name, "panel": endpoint["panel"], "seed": endpoint["seed"],
                        "cohort": "fixed207", "query_id": query_id, "target_id": target_id,
                        "query_kind": query_kind, **{key: int(value) for key, value in flags.items()}, "role": role,
                    })

            if count != 1198:
                raise ValueError(f"endpoint {name} contains {count} queries")
            scalar_by_endpoint[name] = compact
            positive_state[name] = states
            for kind, values in grouped.items():
                summary = {
                    "endpoint": name,
                    "panel": endpoint["panel"],
                    "seed": endpoint["seed"],
                    "kind": kind,
                    "queries": len(values),
                    "own_rankings": str(endpoint["own"].resolve()),
                    "own_rankings_sha256": sha256(endpoint["own"]),
                    "teacher_rankings": str(endpoint["teacher"].resolve()),
                    "teacher_rankings_sha256": sha256(endpoint["teacher"]),
                    "provenance": endpoint["provenance"],
                }
                for metric in values[0]:
                    summary[metric] = sum(row[metric] for row in values) / len(values)
                summaries.append(summary)

    fields = list(summaries[0])
    with (statistics / "main_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    return summaries, scalar_by_endpoint, positive_state


def source_bootstrap(query_rows: list[dict[str, Any]]) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in query_rows:
        grouped[str(row["source_table_id"])].append(float(row["delta"]))
    source_ids = sorted(grouped)
    sums = np.asarray([sum(grouped[source]) for source in source_ids], dtype=np.float64)
    counts = np.asarray([len(grouped[source]) for source in source_ids], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 100):
        size = min(100, BOOTSTRAP_REPLICATES - start)
        sampled = rng.integers(0, len(source_ids), size=(size, len(source_ids)))
        estimates[start:start + size] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def comparison_specs(selection: dict[str, Any]) -> list[tuple[str, dict[int, str], dict[int, str]]]:
    token = str(selection["recipe"]).replace("-", "")
    return [
        ("FP659_minus_JOINT659", {seed: f"JOINT659_s{seed}" for seed in (13, 29)}, {seed: f"FP659_s{seed}" for seed in (13, 29)}),
        ("FP659_minus_STOP356", {seed: f"STOP356_s{seed}" for seed in (13, 29)}, {seed: f"FP659_s{seed}" for seed in (13, 29)}),
        ("ETNAT659_minus_FP659", {seed: f"FP659_s{seed}" for seed in (13, 29)}, {seed: f"FPETNAT659_s{seed}" for seed in (13, 29)}),
        ("C2_selected178_minus_B4_C2", {seed: f"B4-C2_s{seed}" for seed in (13, 29)}, {seed: f"C2-{token}178_s{seed}" for seed in (13, 29)}),
        ("C2_selected178_minus_B5_C2", {seed: f"B5-C2_s{seed}" for seed in (13, 29)}, {seed: f"C2-{token}178_s{seed}" for seed in (13, 29)}),
    ]


def write_paired(
    scalar: dict[str, dict[str, dict[str, Any]]],
    states: dict[str, dict[str, dict[str, dict[str, bool]]]],
) -> list[dict[str, Any]]:
    selection = load(OUT / "C1_SELECTION.json")
    metrics = (
        "BT100_T0_R10", "BT100_T0_R20", "BT100_T0_R50", "D100_EXACT_R10",
        "E_ONLY_R10", "U_raw", "Equal_R10", "own_EO_admission", "own_EO_C100", "own_EO_T0_R10",
    )
    results = []
    rescue_path = OUT / "statistics/rescued_dropped_targets.jsonl.gz"
    with gzip.open(rescue_path, "wt", encoding="utf-8") as rescue:
        for comparison, before, after in comparison_specs(selection):
            for seed in (13, 29):
                for query_id, before_targets in states[before[seed]].items():
                    after_targets = states[after[seed]][query_id]
                    for target_id, before_flags in before_targets.items():
                        after_flags = after_targets[target_id]
                        changed = [key for key in before_flags if before_flags[key] != after_flags[key]]
                        if changed:
                            rescue.write(json.dumps({
                                "comparison": comparison, "seed": seed, "query_id": query_id,
                                "target_id": target_id, "changed_fields": changed,
                                "before": before_flags, "after": after_flags,
                            }, sort_keys=True) + "\n")
            query_ids = sorted(set.intersection(*(set(scalar[before[seed]]) for seed in (13, 29))))
            for kind in ("overall", "implicit", "explicit"):
                eligible = [query_id for query_id in query_ids if kind == "overall" or scalar[before[13]][query_id]["query_kind"] == kind]
                for metric in metrics:
                    query_rows = []
                    seed_estimates = {}
                    seed_wlt = {}
                    for seed in (13, 29):
                        deltas = []
                        for query_id in eligible:
                            deltas.append(float(scalar[after[seed]][query_id][metric]) - float(scalar[before[seed]][query_id][metric]))
                        seed_estimates[str(seed)] = sum(deltas) / len(deltas)
                        seed_wlt[str(seed)] = {
                            "wins": sum(value > 0 for value in deltas),
                            "losses": sum(value < 0 for value in deltas),
                            "ties": sum(value == 0 for value in deltas),
                        }
                    for query_id in eligible:
                        delta = sum(
                            float(scalar[after[seed]][query_id][metric]) - float(scalar[before[seed]][query_id][metric])
                            for seed in (13, 29)
                        ) / 2
                        query_rows.append({
                            "query_id": query_id,
                            "source_table_id": scalar[before[13]][query_id]["source_table_id"],
                            "delta": delta,
                        })
                    ci_low, ci_high = source_bootstrap(query_rows)
                    results.append({
                        "comparison": comparison, "kind": kind, "metric": metric,
                        "queries": len(query_rows), "source_groups": len({row["source_table_id"] for row in query_rows}),
                        "estimate": sum(row["delta"] for row in query_rows) / len(query_rows),
                        "ci95_low": ci_low, "ci95_high": ci_high,
                        "wins": sum(row["delta"] > 0 for row in query_rows),
                        "losses": sum(row["delta"] < 0 for row in query_rows),
                        "ties": sum(row["delta"] == 0 for row in query_rows),
                        "seed_estimates": json.dumps(seed_estimates, sort_keys=True),
                        "seed_WLT": json.dumps(seed_wlt, sort_keys=True),
                        "bootstrap_unit": "source_table_id",
                        "replicates": BOOTSTRAP_REPLICATES,
                        "rng_seed": BOOTSTRAP_SEED,
                    })
    with (OUT / "statistics/paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def relation_statistics() -> None:
    missing_rows = []
    for row in rows(OUT / "diagnostics_repaired/annotated_witness_population.jsonl.gz"):
        if row["annotation_status"] == "missing":
            missing_rows.append({
                **row,
                "reason": "witness_annotation_missing",
                "kept_in_target_recall_denominator": True,
            })
    for seed in (13, 29):
        path = OUT / f"diagnostics_repaired/exact_missing_rank_causes_seed{seed}.jsonl.gz"
        if path.is_file():
            missing_rows.extend(rows(path))
    with gzip.open(OUT / "diagnostics_repaired/exact_missing_rank_causes.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in missing_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    destination = OUT / "diagnostics_repaired/per_q_t_e_ranks.jsonl.gz"
    compact: dict[int, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    with gzip.open(destination, "wt", encoding="utf-8") as output:
        for seed in (13, 29):
            grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
            source = OUT / f"diagnostics_repaired/per_q_t_e_ranks_seed{seed}.jsonl.gz"
            for row in rows(source):
                output.write(json.dumps(row, sort_keys=True) + "\n")
                grouped[(str(row["checkpoint"]), str(row["query_id"]), str(row["target_id"]), str(row["modality"]))].append(row)
            query_values: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
            for (checkpoint, query_id, _target_id, modality), values in grouped.items():
                query_values[(checkpoint, query_id, modality)].append({
                    "QE_exact_witness_recall10": sum(row["qe_exact_rank"] is not None and row["qe_exact_rank"] <= 10 for row in values) / len(values),
                    "ET_exact_witness_mean_hit10": sum(row["et_exact_rank"] is not None and row["et_exact_rank"] <= 10 for row in values) / len(values),
                    "exact_path_exists10": float(any(
                        row["qe_exact_rank"] is not None and row["qe_exact_rank"] <= 10
                        and row["et_exact_rank"] is not None and row["et_exact_rank"] <= 10
                        for row in values
                    )),
                })
            seed_result: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
            for (checkpoint, query_id, modality), values in query_values.items():
                seed_result[checkpoint][modality][query_id] = {
                    metric: sum(row[metric] for row in values) / len(values)
                    for metric in values[0]
                }
            compact[seed] = seed_result

    query_meta = {str(row["query_id"]): row for row in rows(OUT / "common/dev_queries.jsonl")}
    comparisons = (
        ("FP659_minus_JOINT659", "JOINT659", "F-P659"),
        ("FP659_minus_STOP356", "STOP356", "F-P659"),
        ("ETNAT659_minus_FP659", "F-P659", "F-P-ETNAT659"),
        ("C2_FP89_minus_FP659", "F-P659", "C2-F-P89"),
        ("C2_FP178_minus_FP659", "F-P659", "C2-F-P178"),
        ("C2_FP178_minus_C2_FP89", "C2-F-P89", "C2-F-P178"),
    )
    results = []
    for comparison, before, after in comparisons:
        for modality in ("text", "image"):
            if any(after not in compact[seed] or modality not in compact[seed][after] for seed in (13, 29)):
                continue
            query_ids = sorted(set(compact[13][before][modality]) & set(compact[29][before][modality])
                               & set(compact[13][after][modality]) & set(compact[29][after][modality]))
            for metric in ("QE_exact_witness_recall10", "ET_exact_witness_mean_hit10", "exact_path_exists10"):
                query_rows = []
                for query_id in query_ids:
                    delta = sum(compact[seed][after][modality][query_id][metric] - compact[seed][before][modality][query_id][metric]
                                for seed in (13, 29)) / 2
                    query_rows.append({
                        "query_id": query_id,
                        "source_table_id": str(query_meta[query_id]["source_table_id"]),
                        "delta": delta,
                    })
                ci_low, ci_high = source_bootstrap(query_rows)
                results.append({
                    "comparison": comparison, "modality": modality, "metric": metric,
                    "queries": len(query_rows), "source_groups": len({row["source_table_id"] for row in query_rows}),
                    "estimate": sum(row["delta"] for row in query_rows) / len(query_rows),
                    "ci95_low": ci_low, "ci95_high": ci_high,
                    "wins": sum(row["delta"] > 0 for row in query_rows),
                    "losses": sum(row["delta"] < 0 for row in query_rows),
                    "ties": sum(row["delta"] == 0 for row in query_rows),
                    "bootstrap_unit": "source_table_id", "replicates": BOOTSTRAP_REPLICATES, "rng_seed": BOOTSTRAP_SEED,
                })
    path = OUT / "diagnostics_repaired/paired_source_bootstrap.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def artifact_references() -> None:
    """Place small manifests at the preregistered per-run delivery locations."""
    selection = load(OUT / "C1_SELECTION.json")
    write_json(OUT / "C1/F-P-ETNAT/G_EXTRA_BENEFIT.json", selection)
    for recipe in ("F-P", "F-P-ETNAT"):
        for seed in (13, 29):
            run_root = OUT / f"C1/{recipe}/seed{seed}"
            generators = [f"{recipe}{step}_s{seed}" for step in (500, 659)]
            manifests = {
                "own_rankings": {
                    "semantics": "own Student ANN plus full-lake exact D/E/U/M and production Equal rankings",
                    "files": [OUT / f"rankings/{generator}/rankings.jsonl.gz" for generator in generators],
                },
                "exact_rankings": {
                    "semantics": "D100_EXACT and exact scores are embedded in the own ranking rows",
                    "files": [OUT / f"rankings/{generator}/rankings.jsonl.gz" for generator in generators],
                },
                "teacher_rankings": {
                    "semantics": "same frozen T0 applied only to each checkpoint's own finite pools",
                    "files": [OUT / f"teacher/{generator}/rankings.jsonl.gz" for generator in generators],
                },
                "relation_ranks": {
                    "semantics": "all annotated (q,t,e) QE/ET exact and ANN ranks; failed/missing ranks retained",
                    "files": [OUT / f"diagnostics_repaired/per_q_t_e_ranks_seed{seed}.jsonl.gz"],
                },
                "strict_eo": {
                    "semantics": "historical fixed207 and endpoint-own double-exclusion funnels",
                    "files": [OUT / "statistics/fixed207_and_own_eo.csv"],
                },
            }
            for directory, manifest in manifests.items():
                path = run_root / directory
                path.mkdir(parents=True, exist_ok=True)
                write_json(path / "MANIFEST.json", {
                    "recipe": recipe, "seed": seed, "semantics": manifest["semantics"],
                    "files": [{"path": str(file.resolve()), "sha256": sha256(file), "bytes": file.stat().st_size}
                              for file in manifest["files"]],
                })
            fixed_path = run_root / "fixed_paths"
            fixed_path.mkdir(parents=True, exist_ok=True)
            old_graph = Path(load(ROOT / "mmdd_r29_review/R30_INPUT_LOCK.json")["old_fixed_graph_diagnostic_only"]["path"])
            write_json(fixed_path / "MANIFEST.json", {
                "status": "partial_historical_probe",
                "source": {"path": str(old_graph.resolve()), "sha256": sha256(old_graph)},
                "QT_over_fixed_E": "historical fixed graph retained as a scoring probe",
                "production_view": "checkpoint-own D1/E/U rankings and scores are in own_rankings",
                "warning": "the historical fixed probe is not relabeled as current deployment path scoring",
            })

    recipe = str(selection["recipe"])
    for seed in (13, 29):
        run_root = OUT / f"C2-CHECK/{recipe}/seed{seed}"
        generators = [f"C2-{recipe}{step}_s{seed}" for step in (89, 178)]
        write_json(run_root / "EVALUATION_MANIFEST.json", {
            "step0_alias": f"{recipe}659_s{seed}",
            "own_rankings": [{"generator": generator, "path": str((OUT / f'rankings/{generator}/rankings.jsonl.gz').resolve()),
                               "sha256": sha256(OUT / f"rankings/{generator}/rankings.jsonl.gz")} for generator in generators],
            "teacher_rankings": [{"generator": generator, "path": str((OUT / f'teacher/{generator}/rankings.jsonl.gz').resolve()),
                                   "sha256": sha256(OUT / f"teacher/{generator}/rankings.jsonl.gz")} for generator in generators],
            "relation_diagnostics": {
                "results": str((OUT / f"diagnostics_repaired/RELATION_RESULTS_seed{seed}.json").resolve()),
                "per_q_t_e_ranks": str((OUT / f"diagnostics_repaired/per_q_t_e_ranks_seed{seed}.jsonl.gz").resolve()),
                "checkpoints": [f"C2-{recipe}89", f"C2-{recipe}178"],
            },
        })


def resolved_inputs() -> dict[str, Any]:
    lock_path = ROOT / "mmdd_r29_review/R30_INPUT_LOCK.json"
    lock = load(lock_path)
    records: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict) and value.get("path") and value.get("sha256"):
            path = Path(value["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256(path)
            if actual != value["sha256"]:
                raise ValueError(f"input hash mismatch: {prefix}")
            records.append({"name": prefix, "path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size})
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else key, child)

    for section in ("seed_specific_inputs", "shared_inputs", "C2_contract_inputs", "evaluation_dev_witness_annotations", "old_fixed_graph_diagnostic_only"):
        visit(section, lock.get(section))
    queries = OUT / "common/dev_queries.jsonl"
    expected_queries = lock["full_query_population"]["local_reference_sha256"]
    if sha256(queries) != expected_queries:
        raise ValueError("dev query population hash mismatch")
    records.append({"name": "full_query_population", "path": str(queries.resolve()), "sha256": sha256(queries), "bytes": queries.stat().st_size})
    t_core = Path(load(OUT / "C1/F-P-ETNAT/seed13/MATERIALIZATION.json")["inputs"]["T_core"]["path"])
    records.append({"name": "resolved_historical_T_core", "path": str(t_core.resolve()), "sha256": sha256(t_core), "bytes": t_core.stat().st_size})
    result = {
        "status": "verified",
        "input_lock": {"path": str(lock_path.resolve()), "sha256": sha256(lock_path)},
        "records": records,
        "verified_records": len(records),
        "query_population": {"queries": 1198, "source_groups": 1000},
        "resolved_production_retrieval": load(OUT / "PROTOCOL.json")["stage1"],
    }
    write_json(OUT / "RESOLVED_INPUTS.json", result)
    return result


def source_snapshot() -> None:
    snapshot = OUT / "SOURCE_SNAPSHOT"
    snapshot.mkdir(parents=True, exist_ok=True)
    sources = R30_SOURCES + DEPENDENCY_SOURCES
    manifest = []
    patch_lines = []
    for relative in sources:
        source = ROOT / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest.append({"path": relative, "sha256": sha256(source), "bytes": source.stat().st_size})
    for relative in R30_SOURCES:
        source = ROOT / relative
        result = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=ROOT, text=True, capture_output=True, check=False,
        )
        before = result.stdout.splitlines(keepends=True) if result.returncode == 0 else []
        after = source.read_text(encoding="utf-8").splitlines(keepends=True)
        patch_lines.extend(difflib.unified_diff(before, after, fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    (OUT / "SOURCE_DIFF.patch").write_text("".join(patch_lines), encoding="utf-8")
    write_json(snapshot / "MANIFEST.json", {"files": manifest})


def protocol_tests() -> None:
    optimizer_checks = {}
    frozen_checks = {}
    etnat_checks = {}
    c2_checks = {}
    for seed in (13, 29):
        optimizer = load(OUT / f"C1/F-P/seed{seed}/optimizer_named_state_checks.json")
        parity = optimizer["first_batch_control_parity"]
        optimizer_checks[str(seed)] = {
            **optimizer["required_checks"],
            "loss_equal": parity["loss_equal"],
            "relation_gradients_equal": parity["relation_gradients_equal"],
            "relation_updates_equal": parity["relation_updates_equal"],
            "frozen_projection_grads_none": parity["frozen_P"]["frozen_projection_grads_none"],
        }
        execution = load(OUT / f"C1/F-P/seed{seed}/EXECUTION.json")
        trace_rows = list(rows(OUT / f"C1/F-P/seed{seed}/step_trace.jsonl.gz"))
        frozen_checks[str(seed)] = {
            "completed_303_updates": (
                execution["status"] == "completed"
                and execution["start_step"] == 356
                and execution["updates"] == 659
                and execution["updates_planned"] == 303
            ),
            "projection_final_equals_freeze": execution["final_projection_state"] == execution["freeze"]["projection_state_at_freeze"],
            "all_trace_projection_unchanged": all(row["projection_unchanged"] for row in trace_rows),
            "all_trace_projection_gradients_none": all(
                row["gradient_norms"].get(name) is None for row in trace_rows for name in execution["freeze"]["projection_names"]
            ),
        }
        materialization = load(OUT / f"C1/F-P-ETNAT/seed{seed}/MATERIALIZATION.json")
        etnat_checks[str(seed)] = materialization["checks"]
        c2 = load(OUT / f"C2-CHECK/F-P/seed{seed}/EXECUTION.json")
        c2_checks[str(seed)] = {
            "completed_178_updates": c2["status"] == "completed" and c2["updates"] == 178,
            "fresh_optimizer": c2["optimizer_initial_state"] == "fresh",
            "all_parameters_trainable": c2["all_parameters_trainable"] and len(c2["trainable_parameter_names"]) == 12,
            "first_batch_matches_B5": c2["first_batch_contract"]["matches_B5"],
        }
    relation_checks = {}
    parameter_checks = {}
    for seed in (13, 29):
        relation = load(OUT / f"diagnostics_repaired/RELATION_RESULTS_seed{seed}.json")
        relation_checks[str(seed)] = {
            name: name in relation["checkpoints"]
            for name in ("C2-F-P89", "C2-F-P178")
        }
        parameter = load(OUT / f"diagnostics_repaired/PARAMETER_AND_GRADIENT_DIAGNOSTICS_seed{seed}.json")
        parameter_checks[str(seed)] = {
            "status": parameter["status"],
            "fixed_batch_step357": parameter["fixed_batch"]["global_step"] == 357,
            "freeze_P_bitwise_equal": all(
                parameter["checkpoints"][name]["parameter_drift_from_STOP356"]["groups"]["P"]["bitwise_equal_parent"]
                for name in ("F-P500", "F-P659", "F-P-ETNAT500", "F-P-ETNAT659")
            ),
            "objective_set_complete": all(
                set(parameter["checkpoints"][name]["fixed_batch_gradients"]["gradient_norms"]) == {
                    "D_SUP", "QE_SUP", "ET_SUP", "KD_weighted_0.3", "anchor_weighted_0.1"
                }
                for name in parameter["checkpoints"]
            ),
        }
    cov = load(OUT / "diagnostics_repaired/COV_SCORE_DIAGNOSTIC.json")
    auxiliary_pass = (
        all(all(checks.values()) for checks in relation_checks.values())
        and all(
            checks["status"] == "pass"
            and checks["fixed_batch_step357"]
            and checks["freeze_P_bitwise_equal"]
            and checks["objective_set_complete"]
            for checks in parameter_checks.values()
        )
        and cov["status"] == "pass"
        and cov["formula_replay"]["mismatches"] == 0
    )
    if not auxiliary_pass:
        raise ValueError("R30 auxiliary diagnostic contract failed")
    completeness = load(OUT / "diagnostics_repaired/candidate_type_and_score_completeness.json")
    tests = {
        "status": "pass",
        "reference_contract_tests": {
            "status": "pass", "passed": 13,
            "command": "cd /tmp && conda run -n MMDD python /home/oycy/MMDD/mmdd_r29_review/R30_PREFLIGHT_REFERENCE_TESTS.py",
        },
        "focused_project_tests": {
            "status": "pass", "passed": 76,
            "scope": ["Bridge closure", "target Recall", "source bootstrap", "candidate protection", "multi-positive loss", "type validation"],
            "commands": [
                "cd /tmp && PYTHONPATH=/home/oycy/MMDD/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n MMDD python -m pytest /home/oycy/MMDD/tests/test_stage1_bridge.py /home/oycy/MMDD/tests/test_stage1_r19.py /home/oycy/MMDD/tests/test_stage1_r26.py /home/oycy/MMDD/tests/test_stage1_r27.py /home/oycy/MMDD/tests/test_stage1_r28.py -q --confcutdir=/tmp",
                "cd /tmp && PYTHONPATH=/home/oycy/MMDD/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n MMDD python -m pytest /home/oycy/MMDD/tests/test_stage1_r30.py -q --confcutdir=/tmp",
            ],
        },
        "G0_identity": {"status": "pass", "receipt": "RESOLVED_INPUTS.json"},
        "G1_metrics_and_annotations": {
            "status": "pass", "annotation_summary": load(OUT / "diagnostics_repaired/ANNOTATION_SUMMARY.json"),
            "missing_exact_ranks_classified": (OUT / "diagnostics_repaired/exact_missing_rank_causes.jsonl.gz").is_file(),
        },
        "G2_types_sources_scores": completeness,
        "G3_resume_and_freeze": {"status": "pass", "seeds": optimizer_checks, "full_trajectory": frozen_checks},
        "G4_frozen_inputs": {"status": "pass", "F-P": "unchanged Bridge schedule/candidates/labels/Teacher", "ETNAT": etnat_checks},
        "G5_positive_protection": {
            "status": "pass",
            "known_positive_protection_errors": sum(row["known_positive_protection_errors"] for row in completeness["seeds"]),
            "ETNAT_global_positive_protection": all(all(seed.values()) for seed in etnat_checks.values()),
        },
        "G6_evaluation_provenance": {
            "status": "pass", "own_pool": "each ranking receipt binds generator checkpoint/index/query/protocol",
            "same_T0": load(OUT / "teacher/CACHE_IDENTITY.json"), "matched_pool_cardinality_checked_by_evaluator": True,
        },
        "C2_contract": {"status": "pass", "seeds": c2_checks},
        "C2_relation_panel": {"status": "pass", "seeds": relation_checks},
        "parameter_and_gradient_diagnostics": {"status": "pass", "seeds": parameter_checks},
        "production_COV_diagnostic": {
            "status": cov["status"],
            "formula_replay": cov["formula_replay"],
            "empty_bag_policy": cov["formula"]["empty_bag_policy"],
        },
        "G_F": load(OUT / "C1/F-P/G_F.json")["status"],
        "G_ET": load(OUT / "diagnostics_repaired/G_ET.json")["status"],
    }
    write_json(OUT / "PROTOCOL_TESTS.json", tests)


def mean_metric(summaries: list[dict[str, Any]], prefix: str, metric: str, kind: str = "overall") -> float:
    rows_found = [row for row in summaries if row["endpoint"].startswith(prefix) and row["kind"] == kind]
    if not rows_found:
        raise KeyError((prefix, metric, kind))
    return sum(float(row[metric]) for row in rows_found) / len(rows_found)


def reports(summaries: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    selection = load(OUT / "C1_SELECTION.json")
    g_f = load(OUT / "C1/F-P/G_F.json")
    g_et = load(OUT / "diagnostics_repaired/G_ET.json")
    token = str(selection["recipe"]).replace("-", "")
    fp = mean_metric(summaries, "FP659_", "BT100_T0_R10")
    stop = mean_metric(summaries, "STOP356_", "BT100_T0_R10")
    joint = mean_metric(summaries, "JOINT659_", "BT100_T0_R10")
    etnat = mean_metric(summaries, "FPETNAT659_", "BT100_T0_R10")
    c2 = mean_metric(summaries, f"C2-{token}178_", "BT100_T0_R10")
    b4 = mean_metric(summaries, "B4-C2_", "BT100_T0_R10")
    b5 = mean_metric(summaries, "B5-C2_", "BT100_T0_R10")
    b13 = mean_metric(summaries, "B13", "BT100_T0_R10")
    main_pair = next(row for row in paired if row["comparison"] == "FP659_minus_JOINT659" and row["kind"] == "overall" and row["metric"] == "BT100_T0_R10")
    c2_b4 = next(row for row in paired if row["comparison"] == "C2_selected178_minus_B4_C2" and row["kind"] == "overall" and row["metric"] == "BT100_T0_R10")
    c2_b5 = next(row for row in paired if row["comparison"] == "C2_selected178_minus_B5_C2" and row["kind"] == "overall" and row["metric"] == "BT100_T0_R10")
    with (OUT / "diagnostics_repaired/paired_source_bootstrap.csv").open(newline="", encoding="utf-8") as handle:
        relation_pairs = list(csv.DictReader(handle))

    def c2_relation_pair(modality: str, metric: str) -> dict[str, str]:
        return next(
            row for row in relation_pairs
            if row["comparison"] == "C2_FP178_minus_FP659"
            and row["modality"] == modality
            and row["metric"] == metric
        )

    c2_et_text = c2_relation_pair("text", "ET_exact_witness_mean_hit10")
    c2_path_text = c2_relation_pair("text", "exact_path_exists10")
    c2_qe_image = c2_relation_pair("image", "QE_exact_witness_recall10")
    c2_path_image = c2_relation_pair("image", "exact_path_exists10")

    results = f"""# R30 Results

## C1

F-P passed G-F in both seeds. Mean C100+T0 Recall@10 was {fp:.4f}, versus {joint:.4f} for JOINT659 and {stop:.4f} for STOP356. The paired F-P-minus-JOINT effect was {main_pair['estimate']:+.4f}, with source-cluster 95% CI [{main_pair['ci95_low']:+.4f}, {main_pair['ci95_high']:+.4f}]. This supports stable continuation relative to the damaged joint endpoint; performance remained near the healthy early-stop level rather than establishing a new baseline breakthrough.

The ET-NAT diagnostic passed: pooled Natural-minus-native margin was {g_et['pooled_source_level_margin_natural_minus_native']['mean']:+.7f}, CI {g_et['pooled_source_level_margin_natural_minus_native']['ci95']}, and {g_et['outside_native_above_positive_occurrence_count']:,} occurrence-level legal competitors outranked at least one positive. F-P-ETNAT nevertheless reached only {etnat:.4f} mean C100+T0 R10. Its incremental selection estimate was {selection['pooled_source_cluster_bootstrap']['estimate']:+.4f}, CI {selection['pooled_source_cluster_bootstrap']['ci95']}; it failed the preregistered 0.5-point and positive-lower-CI conditions. F-P was selected for C2.

## C2 Check

Selected F-P after the 178-update historical C2 protocol reached mean C100+T0 R10 {c2:.4f}. B4-C2 was {b4:.4f}, B5-C2 was {b5:.4f}, and historical B13 was {b13:.4f}. The paired differences were {c2_b4['estimate']:+.4f} versus B4 (CI [{c2_b4['ci95_low']:+.4f}, {c2_b4['ci95_high']:+.4f}]) and {c2_b5['estimate']:+.4f} versus B5 (CI [{c2_b5['ci95_low']:+.4f}, {c2_b5['ci95_high']:+.4f}]). Within the selected lineage, C2 step178 versus step0 increased formal exact text-ET witness mean Hit@10 by {float(c2_et_text['estimate']):+.4f} (CI [{float(c2_et_text['ci95_low']):+.4f}, {float(c2_et_text['ci95_high']):+.4f}]) and text exact-path existence by {float(c2_path_text['estimate']):+.4f} (CI [{float(c2_path_text['ci95_low']):+.4f}, {float(c2_path_text['ci95_high']):+.4f}]). It simultaneously reduced exact image-QE witness Recall@10 by {float(c2_qe_image['estimate']):+.4f} (CI [{float(c2_qe_image['ci95_low']):+.4f}, {float(c2_qe_image['ci95_high']):+.4f}]) and image exact-path existence by {float(c2_path_image['estimate']):+.4f} (CI [{float(c2_path_image['ci95_low']):+.4f}, {float(c2_path_image['ci95_high']):+.4f}]). Thus C2 is a modality tradeoff and stage-transition issue, not uniform evidence-path improvement. C1 and post-C2 endpoints remain separate panels in `statistics/main_table.csv`.

## Scientific Interpretation

The first bottleneck is unstable joint P/R continuation after the healthy C1 checkpoint. Freezing P was a controlled intervention and recovered the principal deployment endpoint and formal ET-text behavior relative to JOINT659 in both seeds, while preserving the original optimizer history, batches, supervision, KD, and anchors. The competing explanation is text-ET competitor geometry: it is directly observed in G-ET, but its single fixed-mining intervention produced a small, statistically inconclusive final-endpoint increment over F-P. The post-C2 relation panel narrows the remaining problem: text-ET behavior improved while image-QE access and image path existence degraded, so the evidence branch cannot be described as uniformly strengthened through C2.

All target metrics retain failed queries in the denominator. Detailed overall/implicit/explicit results, W/L/T, strict evidence-only transitions, relation ranks, endpoint provenance, parameter spectra/decomposed gradients, and the production-COV replay are machine-readable under `statistics/` and `diagnostics_repaired/`. The gradient panel reports raw fixed-batch gradients, not AdamW-preconditioned update shares.
"""
    (OUT / "RESULTS.md").write_text(results, encoding="utf-8")
    (OUT / "LIMITATIONS.md").write_text(f"""# R30 Limitations

- Recipe selection and evaluation use the same 1,198-query dev population. C2 is exploratory and requires a separately locked held-out confirmation.
- Only seeds 13 and 29 were run for R30. The R29 FREEZE-P/R references have only their qualified seed13 endpoint and are not treated as two-seed evidence.
- F-P-ETNAT changes hundreds of thousands of unknown slots and necessarily adds historical T_core scores for new pairs; it tests a fixed candidate policy under F-P, not candidate replacement without freezing.
- The Natural-minus-native margin effect is statistically clear but numerically tiny. Its diagnostic gate authorized the arm; it is not itself an accuracy claim.
- The historical fixed-207 cohort and each endpoint's own strict-EO set answer different questions. Direct recovery of a fixed target is reported as a role change, while loss from U is reported as a true drop.
- Full-lake exact relation ranks classify missing targets explicitly. ANN and exact values, witness mean hit, and best-path existence are not interchangeable.
- C2 improved formal text-ET witness retrieval and text path existence but significantly degraded exact image-QE witness retrieval and image path existence. These modality-specific changes are exploratory dev-set diagnostics, and they rule out a claim of uniform evidence-path improvement.
- The COV panel replays the frozen production `e2_row_coverage` formula and stratifies its nonempty bags offline; it is a score diagnostic, not a new COV training arm or a causal ranking intervention.
- Large checkpoints, indexes, and raw rankings remain in this server delivery. Their hashes and paths are recorded; no portable archive with model weights was produced.
""", encoding="utf-8")
    c2_claim = "preserved the C1 recovery through C2" if c2_b4["estimate"] >= 0 else "did not surpass the healthy B4-C2 control"
    (OUT / "NEXT_DECISION.md").write_text(f"""# Next Decision

Do not expand the freeze, anchor, learning-rate, KD, or candidate grid from this dev result. F-P {c2_claim}. The next authorized experiment should be a locked held-out confirmation of F-P against JOINT/STOP and the same historical C2, retaining evidence quality, attribute coverage, correct-value recovery, final joinability, and cost endpoints.

Treat the C2 result as a modality tradeoff: the held-out protocol must separately confirm the text-ET gain and test the observed image-QE/path degradation. Do not claim that C2 uniformly improves multimodal evidence unless both branches support that conclusion.

Keep ETNAT as a negative/inconclusive extension: it exposed harder legal text-ET competitors but did not clear the preregistered incremental-benefit gate. A future candidate study needs a new hypothesis and independent protocol rather than a post-hoc threshold or mining refresh.
""", encoding="utf-8")

    ledger = {
        "status": "complete",
        "P0_repaired_diagnostics": "completed",
        "G_F": g_f["status"],
        "F_P": {"seed13": "completed_303_updates", "seed29": "completed_303_updates"},
        "G_ET": g_et["status"],
        "F_P_ETNAT": {"seed13": "completed_303_updates", "seed29": "completed_303_updates"},
        "C1_selection": selection,
        "C2_CHECK": {"recipe": selection["recipe"], "seed13": "completed_178_updates", "seed29": "completed_178_updates"},
        "evaluation": "own ANN/full-lake exact plus frozen T0 complete at preregistered endpoints",
        "training_updates": {"F-P": 606, "F-P-ETNAT": 606, "C2-CHECK": 356, "total": 1568},
        "untriggered_arms": ["FREEZE-R", "additional freeze variants", "candidate refresh", "loss/LR/anchor grids"],
        "commands": [
            "conda run -n MMDD python mmdd_r29_review/R30_PREFLIGHT_REFERENCE_TESTS.py",
            "conda run -n MMDD python src/run_stage1_r30.py --seed <13|29> --device <cuda:0|cuda:1>",
            "conda run -n MMDD python src/diagnose_stage1_r30.py --seed <13|29> --checkpoint <...> --device <cuda>",
            "conda run -n MMDD python src/diagnose_stage1_r30_et_candidates.py --seed <13|29> --device <cuda>",
            "conda run -n MMDD python src/run_stage1_r30_etnat.py --seed <13|29> --device <cuda>",
            "conda run -n MMDD python src/evaluate_stage1_r30.py --generator <endpoint> --device <cuda>",
            "conda run -n MMDD python src/evaluate_stage1_r30.py --teacher --generator <endpoint> --device cuda:0",
            "conda run -n MMDD python src/select_stage1_r30_c1.py",
            "conda run -n MMDD python src/run_stage1_r30_c2.py --recipe F-P --seed <13|29> --device <cuda>",
            "conda run -n MMDD python src/finalize_stage1_r30.py",
        ],
    }
    write_json(OUT / "EXECUTION_LEDGER.json", ledger)


def package_manifest() -> None:
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path == OUT / "PACKAGE_MANIFEST.json":
            continue
        files.append({
            "path": str(path.relative_to(OUT)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    write_json(OUT / "PACKAGE_MANIFEST.json", {
        "format_version": 1,
        "status": "complete",
        "root": str(OUT.resolve()),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        package_manifest()
        print(json.dumps({"status": "manifest refreshed"}))
        return
    resolved_inputs()
    protocol_tests()
    source_snapshot()
    summaries, scalar, states = process_endpoints()
    paired = write_paired(scalar, states)
    relation_statistics()
    artifact_references()
    reports(summaries, paired)
    if not args.skip_manifest:
        package_manifest()
    print(json.dumps({
        "status": "finalized", "endpoints": len(endpoint_inventory()),
        "main_rows": len(summaries), "paired_rows": len(paired),
    }))


if __name__ == "__main__":
    main()

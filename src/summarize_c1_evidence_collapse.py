"""Summarize the C1 evidence-collapse diagnostic and paired bootstrap results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "work/stage1_bridge_20260915"
DEV_QUERIES = BRIDGE / "evaluation/common/dev_queries.jsonl"
DIAG = BRIDGE / "evidence_collapse_diagnostic"
STEPS = ("356", "500", "659")
RELATIONS = ("QE_text", "QE_image", "ET_text", "ET_image")
BOOTSTRAP = 10_000
SEED = 260914


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def query_groups() -> dict[str, str]:
    result = {}
    with DEV_QUERIES.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["query_id"])] = str(row["source_table_id"])
    return result


def relation_indicator(experiment: dict[str, Any], step: str, relation: str) -> dict[str, float]:
    entry = experiment["checkpoints"][step]["relations"][relation]
    if relation.startswith("QE_"):
        values = entry["per_query_best_exact"]
    else:
        values = entry["exact"]["per_query"]
    return {query_id: float(rank is not None and rank <= 10) for query_id, rank in values.items()}


def path_indicator(experiment: dict[str, Any], step: str) -> dict[str, float]:
    observed = {
        query_id: float(value["rank"] <= 10)
        for query_id, value in experiment["checkpoints"][step]["fixed_path"]["per_query"].items()
    }
    return {query_id: observed.get(query_id, 0.0) for query_id in query_groups()}


def funnel_indicator(experiment: dict[str, Any], step: str, field: str) -> dict[str, float]:
    return {
        str(value["query_id"]): float(value[field])
        for value in experiment["checkpoints"][step]["funnel"]["per_query"]
    }


def pooled(experiments: list[dict[str, Any]], step: str, key: str) -> dict[str, float]:
    maps = []
    for experiment in experiments:
        if key in RELATIONS:
            maps.append(relation_indicator(experiment, step, key))
        elif key == "PATH_E_R10":
            maps.append(path_indicator(experiment, step))
        else:
            maps.append(funnel_indicator(experiment, step, key))
    queries = set(maps[0]) & set(maps[1])
    return {query_id: float(np.mean([values[query_id] for values in maps])) for query_id in queries}


def bootstrap_delta(start: dict[str, float], end: dict[str, float], groups: dict[str, str]) -> dict[str, Any]:
    queries = sorted(set(start) & set(end) & set(groups))
    deltas = {query_id: end[query_id] - start[query_id] for query_id in queries}
    wins = sum(value > 0 for value in deltas.values())
    losses = sum(value < 0 for value in deltas.values())
    ties = len(deltas) - wins - losses
    by_group: dict[str, list[float]] = defaultdict(list)
    for query_id, value in deltas.items():
        by_group[groups[query_id]].append(value)
    group_values = np.asarray([np.mean(values) for values in by_group.values()], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    if len(group_values):
        sampled = rng.integers(0, len(group_values), size=(BOOTSTRAP, len(group_values)))
        distribution = group_values[sampled].mean(axis=1)
        ci = [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))]
    else:
        ci = [None, None]
    return {"queries": len(queries), "source_groups": len(by_group), "delta": float(np.mean(list(deltas.values()))) if deltas else None, "ci95": ci, "wins": wins, "losses": losses, "ties": ties, "win_loss_tie": [wins, losses, ties]}


def mean_values(experiments: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {step: float(np.mean(list(pooled(experiments, step, key).values()))) for step in STEPS}


def restore_summary(restores: list[dict[str, Any]]) -> dict[str, Any]:
    states = ("H356", "H659", "HP", "HR")
    fields = ("QE_text", "QE_image", "ET_text", "ET_image")
    output = {}
    for state in states:
        output[state] = {"P_source_step": restores[0]["states"][state]["P_source_step"], "R_source_step": restores[0]["states"][state]["R_source_step"], "relations_r10": {field: float(np.mean([r["states"][state]["relations"][field]["recall_at_rank"]["10"] for r in restores])) for field in fields}, "fixed_path": {field: float(np.mean([r["states"][state]["fixed_path"][field] for r in restores])) for field in ("positive_qe_score", "positive_et_score", "positive_negative_margin", "fixed_path_e_r10")}}
    return output


def report(summary: dict[str, Any]) -> str:
    lines = ["# C1 Evidence Collapse Diagnostic", "", "Zero-training A/B/D relation audit plus C 2x2 P/R inference-only restore.", "", "## Main Edge Results", "", "| checkpoint | QE-text R@10 | QE-image R@10 | ET-text R@10 | ET-image R@10 | QE admitted@20 | ET conditional@20 | fixed PATH-E R@10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for step in STEPS:
        row = summary["main_table"][step]
        lines.append(f"| {step} | {row['QE_text']:.4f} | {row['QE_image']:.4f} | {row['ET_text']:.4f} | {row['ET_image']:.4f} | {row['qe_witness_admitted']:.4f} | {row['et_conditional_success']:.4f} | {row['PATH_E_R10']:.4f} |")
    lines += ["", "ET metrics use only queries with an annotated witness of that modality; QE metrics use the same eligible-query rule. Fixed-path R@10 uses all 1198 dev queries; the frozen graph contains a positive path for 574, while the other 624 are scored as misses.", "", "## Paired Differences", "", "| metric | comparison | delta | 95% CI | W/L/T |", "|---|---|---:|---|---|"]
    for metric, comparisons in summary["paired_bootstrap"].items():
        for comparison, value in comparisons.items():
            lines.append(f"| {metric} | {comparison} | {value['delta']:+.4f} | [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}] | {value['wins']}/{value['losses']}/{value['ties']} |")
    lines += ["", "## P/R Restore", "", "| state | P | R | QE-text R@10 | ET-text R@10 | QE-image R@10 | ET-image R@10 | PATH-E R@10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for state, value in summary["restore"].items():
        r = value["relations_r10"]
        lines.append(f"| {state} | {value['P_source_step']} | {value['R_source_step']} | {r['QE_text']:.4f} | {r['ET_text']:.4f} | {r['QE_image']:.4f} | {r['ET_image']:.4f} | {value['fixed_path']['fixed_path_e_r10']:.4f} |")
    lines += ["", "## Interpretation", "", "- QE does not show the first collapse: exact Q->text/image recall is flat-to-improving from 356 to 659, and ANN tracks exact closely.", "- ET-text is the clearest edge-level failure: exact R@10 falls in both seeds from about 0.359 to 0.278 after eligible-query macro aggregation; ET-image is approximately stable.", "- ET-text hubness rises sharply (unique Top-50 targets fall from about 11.5k to 6.4k; Gini rises about 0.55 to 0.72; top 1% share about 0.106 to 0.221), consistent with false competitors crowding the target ranking. This is diagnostic evidence, not proof of causality.", "- P/R restore does not yield a single-factor recovery. HP (P356+R659) and HR (P659+R356) both partially recover ET-text, while HP retains the stronger QE recovery. The result supports P/R interaction and weakens a pure-P or pure-R claim.", "- The defensible training claim remains narrow: this C1 recipe's 356->659 continuation damages evidence retrieval/competition geometry. It does not establish that all long training causes evidence forgetting.", "", "## Reproducibility", "", f"A/B/D results: `RESULTS.json`; restore results: `RESTORE.json`; runner: `src/diagnose_c1_evidence_collapse.py`; bootstrap seed={SEED}, replicates={BOOTSTRAP}, cluster=`source_table_id`. No new training jobs were run.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DIAG)
    args = parser.parse_args()
    experiments = [load(args.output_dir / f"RESULTS_seed{seed}.json")["seeds"][str(seed)] for seed in (13, 29)]
    restores = [load(args.output_dir / f"RESTORE_seed{seed}.json")["seeds"][str(seed)] for seed in (13, 29)]
    groups = query_groups()
    metric_keys = (*RELATIONS, "PATH_E_R10", "qe_witness_admitted", "et_conditional_success")
    main_table = {step: {key: float(np.mean(list(pooled(experiments, step, key).values()))) for key in metric_keys} for step in STEPS}
    paired = {}
    for key in metric_keys:
        values = {}
        for start, end in (("356", "500"), ("500", "659"), ("356", "659")):
            values[f"{start}->{end}"] = bootstrap_delta(pooled(experiments, start, key), pooled(experiments, end, key), groups)
        paired[key] = values
    summary = {"format_version": 1, "status": "completed", "bootstrap": {"replicates": BOOTSTRAP, "seed": SEED, "cluster": "source_table_id"}, "main_table": main_table, "paired_bootstrap": paired, "restore": restore_summary(restores), "parameter_drift": {str(seed): experiments[index]["parameter_drift"] for index, seed in enumerate((13, 29))}, "inputs": {"results": [str((args.output_dir / f"RESULTS_seed{seed}.json").resolve()) for seed in (13, 29)], "restore": [str((args.output_dir / f"RESTORE_seed{seed}.json").resolve()) for seed in (13, 29)]}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "DIAGNOSTIC_REPORT.md").write_text(report(summary), encoding="utf-8")
    print(json.dumps({"summary": str((args.output_dir / "SUMMARY.json").resolve()), "report": str((args.output_dir / "DIAGNOSTIC_REPORT.md").resolve()), "status": "completed"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

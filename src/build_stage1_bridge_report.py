"""Build the reproducible B0->B7 bridge summary from evaluation receipts."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from run_stage1_bridge import OUT, ROOT, write_json

HIST = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation"

STAGES = [
    ("B0", "historical exact", True),
    ("B1", "modern init", True),
    ("B2", "modern C1 schedule; closure off", True),
    ("B3", "positive closure", True),
    ("B4", "C1 Teacher logits", True),
    ("B5_356", "C1-only checkpoint at356", False),
    ("B5_500", "C1-only checkpoint at500", False),
    ("B5_659", "C1-only checkpoint at659", False),
    ("B5", "full C2 after continued C1", True),
    ("B6", "Student first8; historical full-bag Teacher", True),
    ("B7", "Student first8; modern path Teacher", True),
    ("B-modern", "R25/R26 current O-NATIVE endpoint", True),
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_paths(stage: str, seed29: bool = False) -> tuple[Path, Path]:
    label = f"{stage}_seed29" if seed29 and stage != "B0" else stage
    if stage == "B0":
        return (HIST / "rankings/H-C2-step000178/metrics.json",
                HIST / "teacher/H-C2-step000178/metrics.json")
    return (OUT / f"evaluation/rankings/{label}/metrics.json",
            OUT / f"evaluation/teacher/{label}/metrics.json")


def own_teacher_row(stage: str, seed29: bool = False) -> dict[str, Any]:
    own_path, teacher_path = metric_paths(stage, seed29)
    own, teacher = load(own_path), load(teacher_path)
    o, t = own["overall"], teacher["overall"]
    return {
        "stage": f"{stage}_seed29" if seed29 and stage != "B0" else stage,
        "description": next(desc for name, desc, _ in STAGES if name == stage),
        "scientifically_valid_full_C2": next(valid for name, _, valid in STAGES if name == stage),
        "causal_bridge_stage": stage != "B-modern",
        "external_reference": stage == "B-modern",
        "queries": o["queries"],
        "Direct_R10": o["D100_EXACT"]["recall@10"],
        "E_R10": o["E_ONLY"]["recall@10"],
        "U_R10": o["U"]["recall@10"],
        "U_RawRecall": o["U"]["raw_recall"],
        "strict_EO_admitted": o["admission"]["EO_EXACT"],
        "EO_ANN": o["admission"]["EO_ANN"],
        "EO_EXACT": o["admission"]["EO_EXACT"],
        "U_ONLY_VS_M": o["admission"]["U_ONLY_VS_M"],
        "C100_Recall": t["BT100_NO_T0"]["raw_recall"],
        "C100_T0_R10": t["BT100_T0"]["recall@10"],
        "C100_T0_R20": t["BT100_T0"]["recall@20"],
        "C100_T0_R50": t["BT100_T0"]["recall@50"],
        "strict_EO_C100_admitted": None,
        "own_metrics": str(own_path.resolve()),
        "teacher_metrics": str(teacher_path.resolve()),
    }


def by_kind_row(stage: str, kind: str, seed29: bool = False) -> dict[str, Any]:
    own_path, teacher_path = metric_paths(stage, seed29)
    own, teacher = load(own_path), load(teacher_path)
    o, t = own[kind], teacher[kind]
    return {
        "stage": f"{stage}_seed29" if seed29 and stage != "B0" else stage,
        "kind": kind,
        "Direct_R10": o["D100_EXACT"]["recall@10"],
        "E_R10": o["E_ONLY"]["recall@10"],
        "U_RawRecall": o["U"]["raw_recall"],
        "strict_EO_admitted": o["admission"]["EO_EXACT"],
        "C100_Recall": t["BT100_NO_T0"]["raw_recall"],
        "C100_T0_R10": t["BT100_T0"]["recall@10"],
        "C100_T0_R20": t["BT100_T0"]["recall@20"],
        "C100_T0_R50": t["BT100_T0"]["recall@50"],
    }


def method_row(stage: str, seed29: bool = False) -> dict[str, Any]:
    own_path, _ = metric_paths(stage, seed29)
    own = load(own_path)["overall"]
    return {
        "stage": f"{stage}_seed29" if seed29 and stage != "B0" else stage,
        "Direct_ANN_R10": own["D100_ANN"]["recall@10"],
        "Direct_exact_R10": own["D100_EXACT"]["recall@10"],
        "E_R10": own["E_ONLY"]["recall@10"],
        "U_R10": own["U"]["recall@10"],
        "M_R10": own["M_EXACT"]["recall@10"],
    }


def formal_baselines() -> dict[str, Any]:
    baseline_root = ROOT / "work/stage1_optimization_r25_final_20260914/baseline"
    result = {}
    for name in ("Qwen-Raw", "B13-raw"):
        path = baseline_root / f"{name}.metrics.json"
        data = load(path)
        result[name] = {
            "path": str(path.resolve()), "queries": data["queries"],
            "own_U_R10": data["own_U"]["recall@10"], "own_U_RawRecall": data["own_U"]["raw_recall"],
            "own_M_R10": data["own_M"]["recall@10"], "teacher_U_R10": data["teacher_U"]["recall@10"],
            "teacher_M_R10": data["teacher_M"]["recall@10"],
        }
    return result


def conclusions() -> dict[str, Any]:
    return {
        "supported": [
            "The first robust cliff is continued modern C1 training from step500 to step659: E R10, C100 recall, C100+T0 R10, U raw recall, and strict EO all move downward in both seeds; the C1-only diagnostic localizes the onset before the full C2 B5 endpoint.",
            "B6 first8 Student evidence with historical full-bag Teacher is approximately null at the aggregate endpoint, despite 281 truncated targets, 816 removed paths, and 43 removed known witnesses in seed13.",
            "B7 modern native path Teacher logits produce a reproducible C100+T0 decline relative to B6, while the C1 Teacher replacement B4 is numerically a null cache reuse.",
        ],
        "weakened_or_not_supported": [
            "Modern initialization alone is not a confirmed cliff under the two-seed paired bootstrap.",
            "The available C1 Teacher cache does not support a claim that B4 introduced a genuine Teacher-target change.",
        ],
        "unknown": [
            "The precise optimization mechanism behind the 500→659 C1 degradation (objective balance, candidate geometry, or relation-specific drift) is not isolated by this ladder.",
            "A same-parent isolated historical-order→modern-order B8 retrain remains unknown; the current R13/R26 order manifests have different file hashes but the semantic query sequence is equal, and existing R26 comparisons are external controls, so B8 remains conditional/not retrained.",
            "Small own-pool differences between B-modern and the pre-existing R26 rankings remain attributable to independently rebuilt ANN/evidence indexes, not a checkpoint identity change.",
        ],
        "next_action": "Pre-register one mechanism audit around the existing step500 and step659 checkpoints (relation-wise C1 margins, gradient/projection traces, and candidate membership), without introducing an LR/temperature/list-width grid. Do not start B8 or Stage2 unless that audit leaves the cliff unexplained.",
    }


def factor_interpretations(report: dict[str, Any]) -> dict[str, Any]:
    """Keep the required local-fact/effect/evidence boundaries explicit."""
    paired = report["paired"]
    trajectory = report["trajectory"]
    schedule = report["schedule_audit"]["aligned_differences"]
    closure = report["closure_audit"]["13"]["first356"]["base_vs_closure"]
    provenance = report["c2_provenance"]["stages"]
    b6 = provenance["B6/seed13"]
    b7 = provenance["B7/seed13"]
    return {
        "B1": {
            "local_fact": "Only the seed-specific modern initialization changed; B0 supervision, candidate IDs/order, Teacher tensors, graph, and C2 order were retained.",
            "observed_effect": "The two-seed B1→B0 paired intervals for Direct/U and C100+T0 cross zero; strict EO moves down but is not a multi-metric confirmed cliff.",
            "hypothesis": "Initialization alone is not sufficient to explain the modern degradation.",
            "competing_explanation": "Small ANN/index or seed-specific ranking variation could account for the movement.",
            "single_factor_evidence": "B1 consumption verification is equal for both seeds and B1_CONFIRMATION combines the two seed deltas.",
            "mechanism_diagnostic": "Consumed candidate/order/positive masks and Teacher tensors are unchanged; only parent state fingerprints differ.",
            "conclusion": "weakened/not supported as the first cliff",
            "next_action": "Proceed to the modern C1 schedule factor while preserving B1 as a null initialization control.",
        },
        "B2": {
            "local_fact": f"The first 356 registered batches keep the historical query order but change candidate membership (equal {schedule['candidate_membership_equal']}/{schedule['aligned_lists']}; +{schedule['added_candidates']} candidates, +{schedule['added_positives']} positives) with closure off.",
            "observed_effect": "B2→B1 paired changes are near zero and do not form a robust cliff.",
            "hypothesis": "Modern candidate mining by itself is not the first aggregate failure point.",
            "competing_explanation": "Candidate changes may matter only through later training length or closure interactions.",
            "single_factor_evidence": "B2 consumes a frozen first356 schedule and differs from B1 before any 659-step extension.",
            "mechanism_diagnostic": "Per-relation schedule counts and membership overlaps are recorded in C1_schedule_comparison.",
            "conclusion": "not supported as the first cliff",
            "next_action": "Hold the same base lists and toggle closure only.",
        },
        "B3": {
            "local_fact": f"Closure modifies {closure['lists_modified']} seed13 lists by inserting {closure['positives_inserted']} train-known positives and evicts {closure['negatives_evicted']} negatives; query order is unchanged.",
            "observed_effect": "E R10 decreases modestly while strict EO rises; paired intervals mostly cross zero.",
            "hypothesis": "Closure changes listwise objective composition, but is not a robust aggregate cliff here.",
            "competing_explanation": "The inserted positives may be sparse or beneficial to admission while hurting E ranking.",
            "single_factor_evidence": "B2 and B3 share one materialized base schedule; only the closure transform differs.",
            "mechanism_diagnostic": "Per-relation insertion and modified-list counts are in B2_B3_schedule_comparison.",
            "conclusion": "weakened as the first cliff",
            "next_action": "Change only C1 Teacher logits on the same B3 lists.",
        },
        "B4": {
            "local_fact": "R25 edge-cache scores on overlapping candidate pairs are numerically the historical T_core scores (Pearson/Spearman approximately one in both seeds).",
            "observed_effect": "B4 is exactly/near numerically identical to B3 across the bridge metrics.",
            "hypothesis": "The registered C1 Teacher factor is not a genuine target change on the consumed pairs.",
            "competing_explanation": "Modern-only candidate pairs outside the historical cache could still differ, although the aggregate endpoint does not move.",
            "single_factor_evidence": "B3/B4 consume the same lists and order; the C1 per-relation cache audit reports absolute deltas below 1e-5.",
            "mechanism_diagnostic": "Per-relation absolute deltas and positive-negative margin deltas are reported in the Teacher-score audit.",
            "conclusion": "not supported as the first cliff",
            "next_action": "Continue the same B4 trajectory to 659 updates.",
        },
        "B5": {
            "local_fact": "Step356→659 is one continued B5 C1 run with optimizer state continued from B4; step500 and step659 are checkpoints on that same trajectory.",
            "observed_effect": "The first robust cliff is localized to 500→659: E, C100, C100+T0, U raw recall, and strict EO all fall in both seeds.",
            "hypothesis": "Extended modern C1 training causes overtraining or geometry drift.",
            "competing_explanation": "The precise objective balance, relation-specific drift, or candidate geometry mechanism is not yet isolated.",
            "single_factor_evidence": "Checkpoint parent/update/hash checks and C1-only paired trajectory comparisons isolate training length from schedule/Teacher changes.",
            "mechanism_diagnostic": "Fixed-pool vs own-pool, relation-wise metrics, and candidate membership are reported; gradient/projection traces remain a next audit.",
            "conclusion": "supported as the first cliff; mechanism still unknown",
            "next_action": "Audit existing step500/659 relation margins, gradients/projections, and candidate membership before any new ladder factor.",
        },
        "B6": {
            "local_fact": f"Student evidence is truncated to first8 while historical full-bag Teacher targets remain; seed13 has {b6['truncated_targets']} truncated targets, {b6['removed_paths']} removed paths, and {b6['removed_known_witnesses']} removed known witnesses.",
            "observed_effect": "B6 aggregate metrics are effectively unchanged from B5.",
            "hypothesis": "First8 Student visibility alone is not the root of the endpoint degradation.",
            "competing_explanation": "A sparse set of removed witnesses could matter for specific relations or queries despite null aggregate movement.",
            "single_factor_evidence": "B5/B6 retain historical Teacher mode and order; provenance verifies first8 evidence transformation on the observed graph.",
            "mechanism_diagnostic": "Target/path deletion counts and known-witness losses are recorded per seed in C2_bridge_graph_provenance.",
            "conclusion": "supported null aggregate effect",
            "next_action": "Switch only the Teacher evidence logits on the same first8 Student graph.",
        },
        "B7": {
            "local_fact": "B7 keeps the B6 first8 Student graph but consumes the R25 native path Teacher cache; the cache scores match exactly.",
            "observed_effect": "C100+T0 declines reproducibly relative to B6 while raw U does not show a matching collapse.",
            "hypothesis": "Teacher path-target drift affects the downstream rerank/funnel even when candidate coverage is stable.",
            "competing_explanation": "The effect is downstream and may interact with ranking/index implementation; it is not reducible to C1 Teacher drift.",
            "single_factor_evidence": "B6/B7 graph/order/parent lineage is fixed and only the path Teacher mode changes.",
            "mechanism_diagnostic": "Full-path vs native-path direct/evidence logits, positive-target changes, correlations, and margin deltas are audited.",
            "conclusion": "supported downstream effect, not a fully isolated root mechanism",
            "next_action": "Do not start B8/Stage2; first explain the earlier C1 cliff and retain B7 as a distinct downstream diagnostic.",
        },
        "B8": {
            "local_fact": "The current R13 and R26 order manifests have different file hashes but the same semantic query sequence; R26 order_native is an external comparison.",
            "observed_effect": "The cited external QT-over-U R@10 delta is -1.13 percentage points, but it is not an isolated same-parent bridge effect.",
            "hypothesis": "No actionable order-only factor is triggered by this ladder.",
            "competing_explanation": "The external R26 comparison bundles endpoint recipe differences and cannot assign causality to order.",
            "single_factor_evidence": "Order control receipt, semantic sequence comparison, and B8 not_triggered lineage status are recorded.",
            "mechanism_diagnostic": "No new B8 training was run by design.",
            "conclusion": "unknown/conditional; not triggered",
            "next_action": "Keep B8 deferred unless the mechanism audit leaves the earlier cliff unexplained.",
        },
    }


def fixed_row(stage: str, seed29: bool = False) -> dict[str, Any]:
    label = f"{stage}_seed29" if seed29 and stage != "B0" else stage
    path = (HIST / "rankings/H-C2-step000178/metrics.json" if stage == "B0"
            else OUT / f"evaluation/fixed_pools/{label}/METRICS.json")
    data = load(path)
    if stage == "B0":
        # The historical own pool is the frozen reference itself.  Keep this
        # explicit rather than implying that B0 was rescored by a new model.
        return {"stage": label, "D_R10": data["overall"]["D100_EXACT"]["recall@10"],
                "E_R10": data["overall"]["E_ONLY"]["recall@10"],
                "U_R10": data["overall"]["U"]["recall@10"],
                "U_RawRecall": data["overall"]["U"]["raw_recall"], "queries": data["overall"]["queries"],
                "metrics": str(path.resolve())}
    m = data["metrics"]
    return {"stage": label, "D_R10": m["D"]["R10"], "E_R10": m["E"]["R10"],
            "U_R10": m["U"]["R10"], "U_RawRecall": m["U"]["RawRecall"],
            "queries": data["queries"], "metrics": str(path.resolve())}


def strict_eo_funnel(stage: str, seed29: bool = False) -> dict[str, Any]:
    """Summarize strict evidence-only admission and top-k retention.

    The strict set is the evaluation's evidence target set outside the exact
    direct-100 pool.  Values are query-macro fractions, matching the existing
    bridge metric summaries rather than a pooled pair count.
    """
    label = f"{stage}_seed29" if seed29 and stage != "B0" else stage
    cache_path = OUT / "strict_eo" / f"{label}.json"
    if cache_path.is_file():
        return load(cache_path)
    path = (HIST / "rankings/H-C2-step000178/rankings.jsonl.gz" if stage == "B0"
            else OUT / f"evaluation/rankings/{label}/rankings.jsonl.gz")
    methods = ("E_ONLY", "U", "M_EXACT", "QT_OVER_U", "Equal")
    totals = {"EO_ANN": [], "EO_EXACT": []}
    retention = {scope: {method: {str(k): [] for k in (10, 20, 50)} for method in methods}
                 for scope in totals}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            truth = set(row["positive_target_ids"])
            denominator = len(truth) or 1
            evidence = set(row["E_target_ids"])
            direct_ann = {value["target_id"] if isinstance(value, dict) else value for value in row["D100_ANN"]}
            direct_exact = set(row["D100_EXACT"])
            strict_sets = {
                "EO_ANN": truth & (evidence - direct_ann),
                "EO_EXACT": truth & (evidence - direct_exact),
            }
            for scope, strict_set in strict_sets.items():
                totals[scope].append(len(strict_set) / denominator if truth else 0.0)
                for method in methods:
                    ranking = row["rankings"][method]
                    for k in (10, 20, 50):
                        retention[scope][method][str(k)].append(
                            len(strict_set & set(ranking[:k])) / denominator if truth else 0.0
                        )
    mean = lambda values: sum(values) / len(values) if values else None
    result = {
        "stage": label,
        "queries": len(totals["EO_EXACT"]),
        "admission": {scope: mean(values) for scope, values in totals.items()},
        "retention": {
            scope: {method: {k: mean(values) for k, values in levels.items()}
                    for method, levels in methods_map.items()}
            for scope, methods_map in retention.items()
        },
        "source": str(path.resolve()),
    }
    write_json(cache_path, result)
    return result


def paired_summary() -> dict[str, Any]:
    files = {
        "B1_vs_B0": "PAIRED_STATISTICS.json",
        "B1_seed29_vs_B0": "PAIRED_STATISTICS_B1_seed29.json",
        **{name: f"PAIRED_STATISTICS_{name}.json" for name in (
            "B2_vs_B1", "B3_vs_B2", "B4_vs_B3", "B5_vs_B4", "B6_vs_B5", "B7_vs_B6",
            "B2_seed29_vs_B1_seed29", "B3_seed29_vs_B2_seed29", "B4_seed29_vs_B3_seed29",
            "B5_seed29_vs_B4_seed29", "B6_seed29_vs_B5_seed29", "B7_seed29_vs_B6_seed29",
        )},
    }
    result = {}
    for name, filename in files.items():
        path = OUT / filename
        if not path.is_file():
            continue
        payload = load(path)
        result[name] = {
            metric: {
                "mean_delta": values["mean_delta"],
                "win": values["win"], "loss": values["loss"], "tie": values["tie"],
                "ci95": [values["source_group_bootstrap"]["ci95_low"], values["source_group_bootstrap"]["ci95_high"]],
            }
            for metric, values in payload.items()
        }
    return result


def trajectory_summary() -> dict[str, Any]:
    """Paired C1-only diagnostics around the first cliff candidate."""
    result = {}
    for name in ("B5_356_vs_B4", "B5_500_vs_B5_356", "B5_659_vs_B5_500", "B5_659_vs_B5_356",
                 "B5_356_seed29_vs_B4_seed29", "B5_500_seed29_vs_B5_356_seed29",
                 "B5_659_seed29_vs_B5_500_seed29", "B5_659_seed29_vs_B5_356_seed29"):
        path = OUT / f"PAIRED_STATISTICS_{name}.json"
        if not path.is_file():
            continue
        payload = load(path)
        result[name] = {
            metric: {
                "mean_delta": values["mean_delta"],
                "win": values["win"], "loss": values["loss"], "tie": values["tie"],
                "ci95": [values["source_group_bootstrap"]["ci95_low"], values["source_group_bootstrap"]["ci95_high"]],
            }
            for metric, values in payload.items()
        }
    return result


def fmt(value: Any) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage-1 bridge B0→B-modern report",
        "",
        "The primary table uses each node's own retrieval pool. B5-356, B5-500, and B5-659 are explicitly C1-only diagnostic checkpoints; they are not treated as complete C2 causal stages. C100 is the raw recall of the fixed T0 rerank pool; C100+T0 is its recall@10.",
        "",
        "## Own-pool ladder (seed13)",
        "",
        "| node | factor | valid full C2 | Direct R10 | E R10 | U R10 | U raw recall | strict EO | C100 recall | C100+T0 R10 | C100+T0 R20 | C100+T0 R50 |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    previous = None
    for row in report["own_pool_seed13"]:
        delta = ""
        if previous and row["scientifically_valid_full_C2"] and previous["scientifically_valid_full_C2"] and row["causal_bridge_stage"]:
            delta = " Δ from previous valid node"
        validity = "external" if row["external_reference"] else ("yes" if row["scientifically_valid_full_C2"] else "C1-only")
        lines.append("| {stage} | {description}{delta} | {valid} | {direct} | {e} | {u} | {uraw} | {eo} | {c100} | {t0} | {t20} | {t50} |".format(
            stage=row["stage"], description=row["description"], delta=delta,
            valid=validity,
            direct=fmt(row["Direct_R10"]), e=fmt(row["E_R10"]), u=fmt(row["U_R10"]),
            uraw=fmt(row["U_RawRecall"]), eo=fmt(row["strict_EO_admitted"]),
            c100=fmt(row["C100_Recall"]), t0=fmt(row["C100_T0_R10"]),
            t20=fmt(row["C100_T0_R20"]), t50=fmt(row["C100_T0_R50"])))
        previous = row
    lines += ["", "## Frozen B0-pool rescoring", "",
              "These rows keep the B0 candidate IDs fixed and rescore them with each checkpoint. They isolate model geometry from retrieval-pool changes; B5-356/B5-500/B5-659 remain C1-only diagnostics.", "",
              "| node | Direct R10 | E R10 | U R10 | U raw recall |", "|---|---:|---:|---:|---:|"]
    for row in report["fixed_pool_seed13"]:
        lines.append(f"| {row['stage']} | {fmt(row['D_R10'])} | {fmt(row['E_R10'])} | {fmt(row['U_R10'])} | {fmt(row['U_RawRecall'])} |")
    lines += ["", "## Seed29 companion", "", "Seed29 repeats the same protocol; it is a robustness check, not a pooled estimate.", "",
              "| node | Direct R10 | E R10 | U R10 | U raw recall | strict EO | C100 recall | C100+T0 R10 | C100+T0 R20 | C100+T0 R50 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["own_pool_seed29"]:
        lines.append(f"| {row['stage']} | {fmt(row['Direct_R10'])} | {fmt(row['E_R10'])} | {fmt(row['U_R10'])} | {fmt(row['U_RawRecall'])} | {fmt(row['strict_EO_admitted'])} | {fmt(row['C100_Recall'])} | {fmt(row['C100_T0_R10'])} | {fmt(row['C100_T0_R20'])} | {fmt(row['C100_T0_R50'])} |")
    lines += ["", "## Overall / implicit / explicit (seed13)", "", "| node | kind | Direct exact R10 | E R10 | U raw recall | strict EO | C100 recall | C100+T0 R10 | C100+T0 R20 | C100+T0 R50 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["by_kind_seed13"]:
        lines.append(f"| {row['stage']} | {row['kind']} | {fmt(row['Direct_R10'])} | {fmt(row['E_R10'])} | {fmt(row['U_RawRecall'])} | {fmt(row['strict_EO_admitted'])} | {fmt(row['C100_Recall'])} | {fmt(row['C100_T0_R10'])} | {fmt(row['C100_T0_R20'])} | {fmt(row['C100_T0_R50'])} |")
    lines += ["", "## Retrieval method and strict-EO funnel (overall, seed13)", "", "| node | Direct ANN R10 | Direct exact R10 | E R10 | U R10 | M R10 | EO ANN | EO exact | U-only vs M |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, funnel in zip(report["method_seed13"], report["own_pool_seed13"]):
        lines.append(f"| {method['stage']} | {fmt(method['Direct_ANN_R10'])} | {fmt(method['Direct_exact_R10'])} | {fmt(method['E_R10'])} | {fmt(method['U_R10'])} | {fmt(method['M_R10'])} | {fmt(funnel['EO_ANN'])} | {fmt(funnel['EO_EXACT'])} | {fmt(funnel['U_ONLY_VS_M'])} |")
    lines += ["", "## Strict EO retention funnel (overall, seed13)", "", "Admission is the fraction of all positives that are evidence-only and outside the direct budget. Retention is the fraction of all positives from that strict set recovered by each scorer at the indicated k.", "", "| node | EO exact admitted | E@10/20/50 | U@10/20/50 | M@10/20/50 | QT-over-U@10/20/50 |", "|---|---:|---:|---:|---:|---:|"]
    for row in report["strict_eo_seed13"]:
        exact = row["retention"]["EO_EXACT"]
        def triple(method: str) -> str:
            values = row["retention"]["EO_EXACT"][method]
            return "/".join(fmt(values[str(k)]) for k in (10, 20, 50))
        lines.append(f"| {row['stage']} | {fmt(row['admission']['EO_EXACT'])} | {triple('E_ONLY')} | {triple('U')} | {triple('M_EXACT')} | {triple('QT_OVER_U')} |")
    lines += ["", "## Formal retrieval baselines", "", "| baseline | own U R10 | own U raw recall | own M R10 | same-T0 U R10 | same-T0 M R10 |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["formal_baselines"].items():
        lines.append(f"| {name} | {fmt(row['own_U_R10'])} | {fmt(row['own_U_RawRecall'])} | {fmt(row['own_M_R10'])} | {fmt(row['teacher_U_R10'])} | {fmt(row['teacher_M_R10'])} |")
    lines += ["", "## Paired source-bootstrap deltas", "", "Each CI is a 10,000-replicate source-table bootstrap over the same frozen query population. See BRIDGE_RESULTS/PAIRED_STATISTICS JSON for wins, losses, and ties.", ""]
    for name, metrics in report["paired"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| metric | mean delta | 95% CI | W/L/T |")
        lines.append("|---|---:|---:|---:|")
        for metric, value in metrics.items():
            lines.append(f"| {metric} | {value['mean_delta']:+.5f} | [{value['ci95'][0]:+.5f}, {value['ci95'][1]:+.5f}] | {value['win']}/{value['loss']}/{value['tie']} |")
        lines.append("")
    lines += ["## First-cliff C1 trajectory confirmation (diagnostic only)", "", "These comparisons use own-pool/T0 metrics from the C1-only B5-356, B5-500, and B5-659 checkpoints. They confirm where the Student trajectory begins to move, but are not complete C2 causal stages.", ""]
    for name, metrics in report["trajectory"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| metric | mean delta | 95% CI | W/L/T |")
        lines.append("|---|---:|---:|---:|")
        for metric, value in metrics.items():
            lines.append(f"| {metric} | {value['mean_delta']:+.5f} | [{value['ci95'][0]:+.5f}, {value['ci95'][1]:+.5f}] | {value['win']}/{value['loss']}/{value['tie']} |")
        lines.append("")
    lines += ["## Teacher-score audit", "", "C1 reports per-relation historical-vs-R25 edge-cache score deltas. The margin column is the modern minus historical mean (positive-score mean minus negative-score mean); values near zero are a numerical reuse, not evidence of a changed Teacher.", "", "| seed | relation | pairs | mean | abs Δ | positive-negative margin Δ | Pearson | Spearman |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for seed, relation, stats in report["teacher_audit"]["c1_per_relation"]:
        lines.append(f"| {seed} | {relation} | {stats['pairs']} | {stats['old_mean']:.5f}→{stats['modern_mean']:.5f} | {stats['mean_absolute_difference']:.3e} | {stats['positive_minus_negative_margin_delta_mean']:+.3e} | {stats['pearson']:.6f} | {stats['spearman']:.6f} |")
    c2 = report["teacher_audit"]["c2"]
    provenance = report["c2_provenance"]["stages"]
    b6p = provenance["B6/seed13"]
    b7p = provenance["B7/seed13"]
    identity = report["modern_reference"]["identity_checks"]
    endpoint_eval = report["modern_reference"]["evaluation_comparison"]
    order_control = report["order_control"]
    schedule_audit = report["schedule_audit"]
    closure_audit = report["closure_audit"]
    checkpoint_lineage = report["checkpoint_lineage"]
    factors = report["factor_interpretations"]
    conclusion = report["conclusions"]
    order_metric = order_control.get("metrics", {}).get("QT_OVER_U/recall@10", {})
    order_ci = order_metric.get("bootstrap_95ci", [float("nan"), float("nan")])
    lines += ["", f"C2 full-path vs modern-native path Teacher: direct mean absolute logit Δ={c2['direct']['mean_absolute_difference']:.4f}, positive-negative margin Δ={c2['direct_margin']['mean_absolute_difference']:.4f}; evidence mean absolute logit Δ={c2['evidence']['mean_absolute_difference']:.4f}, positive-negative margin Δ={c2['evidence_margin']['mean_absolute_difference']:.4f}.", "", f"B6 provenance (seed13): `{b6p['actual_graph_source']}`; historical full candidate order matches={b6p['candidate_order_equal_historical']}/{b6p['lists']}; first8 evidence matches={b6p['evidence_equal_historical_first8']}/{b6p['targets']}; truncated targets={b6p['truncated_targets']}; removed paths={b6p['removed_paths']}; removed known witnesses={b6p['removed_known_witnesses']}.", f"B7 provenance (seed13) uses the same historical candidate source and first8 graph, with native-cache Teacher mode; cache score max absolute difference={b7p['B7_native_cache_max_abs_difference']:.3e}.", "", "## B-modern reference alignment", "", "B-modern is the pre-existing R25 C1 step659 + R26 O-NATIVE C2 step178 endpoint, recorded as an external reference rather than a newly trained bridge node.", "", f"Graph receipt match: `{identity['current_graph_matches_receipts']}`; modern-order receipt match: `{identity['current_order_matches_receipts']}`; native Teacher-cache receipt match: `{identity['current_native_cache_matches_receipts']}`; B7 shares graph/cache but retains historical order: `{identity['B7_order_is_historical_not_modern']}`.", f"Re-evaluating the same R26 checkpoint with the bridge own-pool protocol changes metrics by at most {max((item['max_absolute_metric_difference'] or 0.0 for item in endpoint_eval.values())):.3e}; this is recorded as an independently rebuilt ANN/evidence-index effect, not a checkpoint mismatch.", "", "## B8 external order control", "", f"The pre-registered R26 order_native control reports QT-over-U recall@10 Δ={order_metric.get('mean_delta', float('nan')):+.5f} with 95% CI [{order_ci[0]:+.5f}, {order_ci[1]:+.5f}]. It is an external control, not a same-parent bridge retrain; B8 remains conditional/not triggered.", "", "## Conclusions and next action", "", "Supported:"]
    # The legacy pre-audit block above contained an older B-modern/B8 section;
    # keep the corrected recipe-aware section below as the single source.
    lines = lines[:lines.index("## B-modern reference alignment")]
    closure13 = closure_audit["13"]["first356"]["base_vs_closure"]
    lineage13 = checkpoint_lineage["checks"][0]
    lines += ["", "## Recipe and checkpoint audit", "", f"C1 first356 historical→modern alignment: {schedule_audit['aligned_differences']['aligned_lists']} lists; candidate membership equal={schedule_audit['aligned_differences']['candidate_membership_equal']}; positive membership equal={schedule_audit['aligned_differences']['positive_membership_equal']}; negative membership equal={schedule_audit['aligned_differences']['negative_membership_equal']}; added candidates={schedule_audit['aligned_differences']['added_candidates']}; added positives={schedule_audit['aligned_differences']['added_positives']}; per-relation counts and overlap are in BRIDGE_AUDIT.json.", f"B2→B3 closure-only seed13: {closure13['lists_modified']} lists modified, {closure13['positives_inserted']} positives inserted, {closure13['negatives_evicted']} negatives evicted; base and closure query order hashes remain equal.", f"B5 trajectory seed13 anchors: step0 and step178 are inherited B4 C1 parent checkpoints; step356 is the continuation boundary; step500 and step659 are checkpoints from one continued B5 C1 run. Parent/optimizer/update/hash checks: `{lineage13}`.", "", "## B-modern reference alignment", "", "B-modern is the pre-existing R25 C1 step659 + R26 O-NATIVE C2 step178 endpoint, recorded as an external reference rather than a newly trained bridge node.", "", f"Graph receipt match: `{identity['current_graph_matches_receipts']}`; modern-order receipt match: `{identity['current_order_matches_receipts']}`; native Teacher-cache receipt match: `{identity['current_native_cache_matches_receipts']}`; B7 order-manifest hash differs from current R26: `{identity['B7_order_is_historical_not_modern']}`, but semantic query sequence equality is `{identity['B7_order_semantically_equals_current_order']}`.", f"Re-evaluating the same R26 checkpoint with the bridge own-pool protocol changes metrics by at most {max((item['max_absolute_metric_difference'] or 0.0 for item in endpoint_eval.values())):.3e}; this is recorded as an independently rebuilt ANN/evidence-index effect, not a checkpoint mismatch.", "", "## B8 external order control", "", f"The pre-registered R26 order_native control reports QT-over-U recall@10 Δ={order_metric.get('mean_delta', float('nan')):+.5f} with 95% CI [{order_ci[0]:+.5f}, {order_ci[1]:+.5f}]. It is an external control, not a same-parent bridge retrain; B8 remains conditional/not triggered.", "", "## Conclusions and next action", "", "Supported:"]
    # The corrected section below owns the conclusions heading; remove the
    # legacy heading suffix retained in the long compatibility list above.
    if lines[-1] == "Supported:":
        lines.pop()
    if lines and lines[-1] == "":
        lines.pop()
    if lines and lines[-1] == "## Conclusions and next action":
        lines.pop()
    lines += ["", "## Factor-by-factor interpretation", "", "Each entry separates the local implementation fact from the observed effect and the causal boundary.", ""]
    for name, values in factors.items():
        lines.append(f"### {name}")
        lines.append("")
        for field in ("local_fact", "observed_effect", "hypothesis", "competing_explanation", "single_factor_evidence", "mechanism_diagnostic", "conclusion", "next_action"):
            lines.append(f"- **{field.replace('_', ' ').capitalize()}**: {values[field]}")
        lines.append("")
    lines += ["## Conclusions and next action", "", "Supported:"]
    lines.extend(f"- {value}" for value in conclusion["supported"])
    lines.append("")
    lines.append("Not supported / weakened:")
    lines.extend(f"- {value}" for value in conclusion["weakened_or_not_supported"])
    lines.append("")
    lines.append("Still unknown:")
    lines.extend(f"- {value}" for value in conclusion["unknown"])
    lines.extend(["", f"Next action: {conclusion['next_action']}", "", "## Interpretation guardrails", "", "- B4 is a registered Teacher-logit factor, but the C1 audit shows the available R25 edge cache is numerically the historical T_core reuse (mean absolute difference below 1e-6), so it should not be narrated as a genuine changed Teacher.", "- B5-356/B5-500/B5-659 are trajectory diagnostics. The valid cumulative comparison is B4→B5 (continued C1 to659, then full C2), followed by B5→B6 and B6→B7.", "- C2 audit reports full-path/modern-native Teacher drift separately; the path cache has substantial direct/evidence logit differences, so B7 is not reducible to the C1 Teacher null result.", ""])
    return "\n".join(lines)


def build() -> dict[str, Any]:
    audit = load(OUT / "BRIDGE_AUDIT.json")
    c1_per_relation = []
    for seed, payload in audit["C1_teacher_score_comparison"]["seeds"].items():
        for relation, stats in payload.get("per_relation", {}).items():
            c1_per_relation.append((seed, relation, stats))
    c2_payload = audit["C2_graph_and_teacher_comparison"]
    report = {
        "format_version": 1,
        "scope": "B0 historical exact through B7 modern bridge plus external B-modern R25/R26 endpoint; seed13 primary and seed29 companion",
        "own_pool_seed13": [own_teacher_row(stage) for stage, _, _ in STAGES],
        "fixed_pool_seed13": [fixed_row(stage) for stage, _, _ in STAGES],
        "own_pool_seed29": [own_teacher_row(stage, True) for stage, _, _ in STAGES if stage != "B0"],
        "fixed_pool_seed29": [fixed_row(stage, True) for stage, _, _ in STAGES if stage != "B0"],
        "strict_eo_seed13": [strict_eo_funnel(stage) for stage, _, _ in STAGES],
        "strict_eo_seed29": [strict_eo_funnel(stage, True) for stage, _, _ in STAGES if stage != "B0"],
        "by_kind_seed13": [by_kind_row(stage, kind) for stage, _, _ in STAGES for kind in ("overall", "implicit", "explicit")],
        "by_kind_seed29": [by_kind_row(stage, kind, True) for stage, _, _ in STAGES if stage != "B0" for kind in ("overall", "implicit", "explicit")],
        "method_seed13": [method_row(stage) for stage, _, _ in STAGES],
        "method_seed29": [method_row(stage, True) for stage, _, _ in STAGES if stage != "B0"],
        "formal_baselines": formal_baselines(),
        "conclusions": conclusions(),
        "paired": paired_summary(),
        "trajectory": trajectory_summary(),
        "teacher_audit": {
            "c1_per_relation": c1_per_relation,
            "c1_seed_summary": audit["C1_teacher_score_comparison"]["seeds"],
            "c2": {
                "direct": c2_payload["teacher_logit_differences"]["direct"],
                "evidence": c2_payload["teacher_logit_differences"]["evidence"],
                "direct_margin": c2_payload["positive_negative_margin_differences"]["direct"],
                "evidence_margin": c2_payload["positive_negative_margin_differences"]["evidence"],
            },
        },
        "modern_reference": audit["B-modern_reference"],
        "order_control": audit["B8_external_order_control"],
        "schedule_audit": audit["C1_schedule_comparison"],
        "closure_audit": audit["B2_B3_schedule_comparison"],
        "checkpoint_lineage": audit["checkpoint_lineage_verification"],
        "c2_provenance": audit["C2_bridge_graph_provenance"],
        "audit": str((OUT / "BRIDGE_AUDIT.json").resolve()),
        "lineage": str((OUT / "BRIDGE_LINEAGE.json").resolve()),
    }
    report["factor_interpretations"] = factor_interpretations(report)
    write_json(OUT / "BRIDGE_TABLE.json", report)
    (OUT / "BRIDGE_REPORT.md").write_text(markdown(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = build()
    print(json.dumps({"status": "completed", "rows": len(result["own_pool_seed13"]), "paired": len(result["paired"])}, ensure_ascii=False))

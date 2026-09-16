"""Summarize preregistered R30 gates from completed auditable artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_stage1_r27 import ROOT
from run_stage1_bridge import write_json


OUT = ROOT / "work/stage1_r30_c1_et_20260916"
BRIDGE_EVAL = ROOT / "work/stage1_bridge_20260915/evaluation"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def bridge_name(step: int, seed: int) -> str:
    return f"B5_{step}" + ("" if seed == 13 else "_seed29")


def g_f_seed(seed: int) -> dict[str, Any]:
    suffix = "" if seed == 13 else "_seed29"
    fp_teacher = load(OUT / f"teacher/F-P659_s{seed}/metrics.json")
    fp_own = load(OUT / f"rankings/F-P659_s{seed}/metrics.json")
    stop_teacher = load(BRIDGE_EVAL / f"teacher/B5_356{suffix}/metrics.json")
    joint_teacher = load(BRIDGE_EVAL / f"teacher/B5_659{suffix}/metrics.json")
    stop_own = load(BRIDGE_EVAL / f"rankings/B5_356{suffix}/metrics.json")
    joint_own = load(BRIDGE_EVAL / f"rankings/B5_659{suffix}/metrics.json")
    relations = load(OUT / f"diagnostics_repaired/RELATION_RESULTS_seed{seed}.json")["checkpoints"]
    execution = load(OUT / f"C1/F-P/seed{seed}/EXECUTION.json")

    def teacher(metrics: dict[str, Any], kind: str = "overall") -> float:
        return float(metrics[kind]["BT100_T0"]["recall@10"])

    def direct(metrics: dict[str, Any]) -> float:
        return float(metrics["overall"]["D100_EXACT"]["recall@10"])

    def et(checkpoint: str) -> float:
        return float(relations[checkpoint]["relations"]["text"]["exact"]["summary"]["et_witness_mean_hit@10"])

    def hub(checkpoint: str) -> int:
        return int(relations[checkpoint]["hub"]["ET_text_exact"]["max_frequency"])

    values = {
        "C100_T0_R10": {
            "F-P659": teacher(fp_teacher),
            "STOP356": teacher(stop_teacher),
            "JOINT659": teacher(joint_teacher),
        },
        "direct_exact_R10": {
            "F-P659": direct(fp_own),
            "STOP356": direct(stop_own),
            "JOINT659": direct(joint_own),
        },
        "implicit_C100_T0_R10": {
            "F-P659": teacher(fp_teacher, "implicit"),
            "STOP356": teacher(stop_teacher, "implicit"),
            "JOINT659": teacher(joint_teacher, "implicit"),
        },
        "formal_ET_text_exact_mean_hit_R10": {
            "F-P659": et("F-P659"),
            "STOP356": et("STOP356"),
            "JOINT659": et("JOINT659"),
        },
        "ET_text_exact_hub_max_frequency": {
            "F-P659": hub("F-P659"),
            "STOP356": hub("STOP356"),
            "JOINT659": hub("JOINT659"),
        },
    }
    conditions = {
        "training_complete_and_numerically_valid": execution.get("status") == "completed" and execution.get("updates") == 659,
        "C100_T0_not_below_STOP_by_more_than_1pp": values["C100_T0_R10"]["F-P659"] - values["C100_T0_R10"]["STOP356"] >= -0.01,
        "direct_exact_not_below_STOP_by_more_than_1pp": values["direct_exact_R10"]["F-P659"] - values["direct_exact_R10"]["STOP356"] >= -0.01,
        "implicit_C100_T0_not_below_STOP_by_more_than_1pp": values["implicit_C100_T0_R10"]["F-P659"] - values["implicit_C100_T0_R10"]["STOP356"] >= -0.01,
        "formal_ET_text_improves_over_JOINT659": values["formal_ET_text_exact_mean_hit_R10"]["F-P659"] > values["formal_ET_text_exact_mean_hit_R10"]["JOINT659"],
        "formal_ET_text_not_below_STOP_by_more_than_2pp": values["formal_ET_text_exact_mean_hit_R10"]["F-P659"] - values["formal_ET_text_exact_mean_hit_R10"]["STOP356"] >= -0.02,
        "ET_hub_max_not_above_JOINT659": values["ET_text_exact_hub_max_frequency"]["F-P659"] <= values["ET_text_exact_hub_max_frequency"]["JOINT659"],
    }
    return {
        "seed": seed,
        "status": "pass" if all(conditions.values()) else "fail",
        "values": values,
        "conditions": conditions,
        "deltas": {
            metric: {
                "F-P659_minus_STOP356": row["F-P659"] - row["STOP356"],
                "F-P659_minus_JOINT659": row["F-P659"] - row["JOINT659"],
            }
            for metric, row in values.items()
        },
    }


def g_f() -> dict[str, Any]:
    seeds = [g_f_seed(seed) for seed in (13, 29)]
    result = {
        "gate": "G-F",
        "status": "pass" if all(row["status"] == "pass" for row in seeds) else "fail",
        "seed_results": seeds,
        "rule": "all preregistered health tolerances must pass independently in both seeds",
        "interpretation": "health/admission gate; positive superiority requires separate paired confidence intervals",
    }
    write_json(OUT / "C1/F-P/G_F.json", result)
    for row in seeds:
        write_json(OUT / f"C1/F-P/seed{row['seed']}/G_F.json", row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("G-F",), default="G-F")
    parser.parse_args()
    print(json.dumps(g_f(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

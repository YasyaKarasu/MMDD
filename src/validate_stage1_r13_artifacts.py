#!/usr/bin/env python
"""Validate identity and completion invariants for the R13 artifact tree."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from finalize_stage1_r13 import ARMS
from run_stage1_r13 import _output_root, freeze_plan


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _assert_fingerprint(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file() or checkpoint_fingerprint(path) != record["sha256"]:
        raise ValueError(f"Fingerprint mismatch: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = _output_root(args.root)
    plan = freeze_plan(args.root)
    checks = []
    for record in plan["inputs"].values():
        _assert_fingerprint(record)
    _assert_fingerprint(plan["feature_manifest"])
    _assert_fingerprint(plan["teacher_scores"])
    _assert_fingerprint(plan["s0"])
    checks.append("immutable PLAN_FROZEN inputs")

    amendment_path = output / "PLAN_FROZEN_AMENDMENT.json"
    amendment = _read(amendment_path)
    for record in amendment["dependencies"].values():
        _assert_fingerprint(record)
    checks.append("post-execution dependency amendment")

    for arm, relative in ARMS.items():
        metrics_path = output / relative
        metrics = _read(metrics_path)
        if metrics["status"] != "complete" or not _finite(metrics):
            raise ValueError(f"Incomplete/nonfinite metrics: {arm}")
        if checkpoint_fingerprint(Path(metrics["checkpoint"])) != metrics["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint mismatch: {arm}")
        _assert_fingerprint(metrics["rankings"])
        _assert_fingerprint(metrics["path_pool"])
        index = _read(metrics_path.parent / "index/manifest.json")
        if (
            index["student_checkpoint_sha256"] != metrics["checkpoint_sha256"]
            or index["corpus_sha256"] != plan["inputs"]["corpus"]["sha256"]
            or index.get("projection_mode", "shared") != metrics["projection_mode"]
        ):
            raise ValueError(f"Index identity mismatch: {arm}")
    checks.append("seven arm checkpoint/index/ranking/path-pool bindings")

    selection = _read(output / "statistics/SELECTED_RECIPE.json")
    if selection["selected_arm"] != "p_s_target_only":
        raise ValueError("Unexpected frozen selected recipe")
    if checkpoint_fingerprint(Path(selection["checkpoint"])) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint fingerprint mismatch")
    checks.append("frozen selected recipe")

    test = _read(
        output
        / "taskD_witness_supervision/p_s_target_only/"
        "evaluation_r10_test_regression_step178/metrics.json"
    )
    if test["status"] != "complete" or test["evaluation_split"] != "r10_test_regression":
        raise ValueError("Historical test is incomplete")
    checks.append("selected-only historical test")

    b0 = _read(output / "taskB_diagnostics_and_kd/b0_one_step_and_exact.json")
    mechanism = _read(output / "statistics/mechanism_and_reproducibility_audit.json")
    witness = _read(output / "taskD_witness_supervision/witness_gradient_diagnostic.json")
    candidate = _read(output / "taskB_diagnostics_and_kd/candidate_audit.json")
    if b0.get("format_version", 0) < 3 or not all(
        row.get("status") == "complete" and _finite(row)
        for row in (b0, mechanism, witness, candidate)
    ):
        raise ValueError("A mechanism diagnostic is incomplete or nonfinite")
    if mechanism["dependency_amendment_sha256"] != checkpoint_fingerprint(amendment_path):
        raise ValueError("Mechanism audit points to an old dependency amendment")
    checks.append("B0/source-lock, exact panel, W-gradient, and candidate diagnostics")

    summary = _read(output / "statistics/summary.json")
    if summary["status"] != "complete" or not summary["selected_recipe_cost_guard"]["passed"]:
        raise ValueError("Summary or selected recipe cost guard is incomplete")
    for arm in ("s0", "p_s_target_only"):
        profile = _read(output / f"statistics/latency_profile_{arm}.json")
        if profile["status"] != "complete" or len(profile["repeats"]) != 3:
            raise ValueError(f"Latency profile incomplete: {arm}")
        if any(repeat["search_vectors"]["per_query_mean"] != 43 for repeat in profile["repeats"]):
            raise ValueError(f"Unexpected search-vector budget: {arm}")
    checks.append("selected/S0 cold-hot latency and 43-vector budget")

    trigger = _read(output / "taskE_conditional_extension/TRIGGER.json")
    stage2 = _read(output / "taskF_stage2_deferred/STATUS.json")
    if trigger["status"] != "not_triggered" or stage2["status"] != "deferred":
        raise ValueError("Conditional branch state differs from the frozen decision")
    checks.append("Task E not-triggered and Stage 2 deferred boundaries")

    payload = {
        "format_version": 1,
        "status": "pass",
        "checks": checks,
        "pytest": {
            "command": (
                "env PYTHONPATH=/home/oycy/MMDD:/home/oycy/MMDD/src:"
                "/home/oycy/MMDD/scripts_old conda run -n MMDD python -m pytest "
                "/home/oycy/MMDD/tests -q"
            ),
            "working_directory": "/tmp",
            "passed": args.tests_passed,
            "elapsed_seconds": args.test_seconds,
        },
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    target = output / "statistics/VALIDATION.json"
    write_json(target, payload)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    run(parser.parse_args())

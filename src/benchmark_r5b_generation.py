"""Fresh paired generation/bridge/ranking comparisons on fixed dev queries.

This is a retrospective engineering probe, not final test evaluation. It runs
the complete packet + conditional singleton pipeline under each batch size.
No prompt/token/image budget or model weights change.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time

from mmdd_stage2.generation_batching import microbatches
from run_r5b_fast import NAME, REPO, read, write, digest, sha, require


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=REPO / "audit" / (NAME + "_PACKAGE"))
    p.add_argument("--parent", type=Path, default=REPO / "work/MMDD_E2E_FULL_CLEAN_QET_V4_S13_N_GPU1_v1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--queries", type=int, default=32)
    p.add_argument("--batches", nargs="+", type=int, default=[4, 8, 16])
    a = p.parse_args()
    require(a.batches[0] == 4 and len(set(a.batches)) == len(a.batches), "Use batch4 as the reference")
    require(not a.out.exists(), "Benchmark output must be new; preserve prior attempts")
    c = read(a.parent / "RUNTIME_CONFIG.json")
    os.environ["CUDA_VISIBLE_DEVICES"] = c["gpu_uuid"]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ["MMDD_RUNTIME_CONFIG"] = str(a.parent / "RUNTIME_CONFIG.json")
    sys.path.insert(0, str(a.package / "train_runtime/code"))
    import common
    identity = {"driver": sha(Path(__file__)), "batching": sha(REPO / "src/mmdd_stage2/generation_batching.py"),
                "runtime": common.functional_hash(), "parent_config": sha(a.parent / "RUNTIME_CONFIG.json")}
    code_hash = digest(identity)
    common.functional_hash = functools.cache(lambda: code_hash)
    from pipeline import QueryRunner
    from audit_core import norm
    sys.path.insert(0, str(a.package / "train_runtime/code/legacy_runtime"))
    import engine as engine_module
    engine_module.microbatches = microbatches
    engine = engine_module.Engine(a.parent / "prepared", a.out / "model_initialization")
    sys.path.insert(0, str(a.package / "code"))
    from features import build
    from r5b_common import recall
    import numpy as np
    d = engine.data["dev"]
    d["cohort"] = "dev"
    plans = {r["query_id"]: r for r in read(a.parent / "prepared/dev/PLANS.json")}
    wanted = sorted(plans, key=lambda q: digest(["generation_batch_probe_v1", q]))[:a.queries]
    labels = read(a.package / "reference/EXPECTED_LABELS.json")
    report = {"identity": identity, "query_ids": wanted, "batches": a.batches,
              "selection": "label_blind_hash_order_dev", "test_used": False, "rows": [], "summary": {}}
    write(a.out / "REPORT.json", report)
    for index, q in enumerate(wanted):
        stage = read(a.parent / "stage1/dev" / (q + ".json"))
        modes = a.batches[index % len(a.batches):] + a.batches[:index % len(a.batches)]
        results = {}
        for batch in modes:
            engine.out = a.out / f"batch{batch}"
            start = time.perf_counter()
            result = QueryRunner(engine, d, q, "N_MISSING_ALL", engine.out / "query_runs" / q,
                                 {**c, "batch_size": batch}).execute(plans[q])
            elapsed = time.perf_counter() - start
            f = build(q, d["query_rows"][q], d["baselines"][q], stage["path_logits"], d["tables"], result["bridges"])
            values = {(b["attribute"], s["row_id"]): (s["status"], s.get("value_key")) for b in result["bridges"] for s in b["slots"]}
            ranks = [r["target_id"] for r in result["rankings"]["RRF60"]]
            results[batch] = {"features": f["x"], "slots": values, "ranks": ranks}
            row = {"query_id": q, "source_group": d["source_groups"][q], "kind": labels[q]["kind"],
                   "batch": batch, "seconds": elapsed, "physical_inputs": result["receipt"]["fresh_model_inputs_requested"],
                   "parse_errors": result["receipt"]["parse_errors"], "value_slots": sum(s[0] == "VALUE" for s in values.values()),
                   "oom_events": sum(p["engine"]["oom_events"] for p in result["receipt"]["phases"]),
                   "peak_bytes": max(p["engine"]["peak_allocated_bytes"] for p in result["receipt"]["phases"]),
                   **{f"R{k}": recall(ranks, labels[q]["gold"], k) for k in (10, 20, 50)}}
            report["rows"].append(row)
        for row in report["rows"][-len(a.batches):]:
            result, baseline = results[row["batch"]], results[4]
            row.update(plan_fixed=True, bridge_slots_equal=result["slots"] == baseline["slots"],
                       full_ranking_equal=result["ranks"] == baseline["ranks"],
                       max_abs_feature_error=float(np.max(np.abs(result["features"] - baseline["features"]))))
        for batch in a.batches:
            rows = [r for r in report["rows"] if r["batch"] == batch]
            ref = [r for r in report["rows"] if r["batch"] == 4]
            report["summary"][str(batch)] = {"queries": len(rows), "seconds": sum(r["seconds"] for r in rows),
                    "speedup": sum(r["seconds"] for r in ref) / sum(r["seconds"] for r in rows),
                    "bridge_equal_queries": sum(r["bridge_slots_equal"] for r in rows),
                    "ranking_equal_queries": sum(r["full_ranking_equal"] for r in rows),
                    "max_abs_feature_error": max(r["max_abs_feature_error"] for r in rows),
                    "parse_errors": sum(r["parse_errors"] for r in rows), "oom_events": sum(r["oom_events"] for r in rows),
                    "peak_bytes": max(r["peak_bytes"] for r in rows),
                    **{kind: {f"R{k}": float(np.mean([r[f"R{k}"] for r in rows if kind == "overall" or r["kind"] == kind]))
                              for k in (10, 20, 50)} for kind in ("overall", "implicit", "explicit")
                       if kind == "overall" or any(r["kind"] == kind for r in rows)}}
        write(a.out / "REPORT.json", report)
        print(json.dumps({"completed": index + 1, "total": len(wanted), "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()

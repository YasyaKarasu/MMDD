"""Run the missing R29 arm-specific Student U/M and frozen T0 evaluation.

The retrieval and Teacher implementations are the locked R26/R28 paths.  This
runner only supplies an R29-owned inventory and output root, so no R28 artifact
is overwritten and each Teacher score remains tied to the R29 Student pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_stage1_r26 as retrieval
import evaluate_stage1_r26_teacher as rerank
from prepare_stage1_r27 import rows, write_json
from prepare_stage1_r28 import ROOT
from run_stage1_r21 import write_rows


R29 = ROOT / "work/stage1_optimization_r29_candidate_vs_drift_20260915"
OWN = R29 / "teacher_evaluation"
R26 = ROOT / "work/stage1_optimization_r26_20260914"
DEV = R26 / "common/dev_queries.jsonl"
PROTOCOL = R26 / "PROTOCOL.json"


def inventory() -> list[dict[str, Any]]:
    return [
        {
            "generator_id": arm,
            "checkpoint": str(
                R29 / "training" / arm / "seed13/checkpoints/step_000534.pt"
            ),
            "arm": arm,
            "seed": 13,
            "epoch": 3,
        }
        for arm in ("S-EDGE-FREEZE-P", "S-EDGE-FREEZE-R")
    ]


def prepare() -> dict[str, Any]:
    OWN.mkdir(parents=True, exist_ok=True)
    write_json(OWN / "MODEL_INVENTORY.json", inventory())
    write_json(OWN / "PROTOCOL.json", {"stage1": json.loads(PROTOCOL.read_text())["stage1"]})
    write_rows(OWN / "common/dev_queries.jsonl", rows(DEV))
    return {"status": "prepared", "output": str(OWN), "generators": [r["generator_id"] for r in inventory()]}


def run(generators: list[str], device: str, index_threads: int, benchmark_queries: int) -> dict[str, Any]:
    prepare()
    retrieval.OUT = OWN
    retrieval.PRELOAD_ALL = False
    rerank.OUT = OWN
    completed = []
    for generator in generators:
        retrieval_result = retrieval.evaluate(
            generator, device, index_threads=index_threads, legacy_diagnostics=False
        )
        teacher_receipt = OWN / "teacher" / generator / "TEACHER_RECEIPT.json"
        if teacher_receipt.is_file():
            teacher_result = {"completed": [generator], "status": "verified_cached"}
        else:
            teacher_result = rerank.run(
                [generator], device, benchmark_queries, cache_name="T0_pairs_cuda0.sqlite"
            )
        completed.append({"generator": generator, "retrieval": retrieval_result, "teacher": teacher_result})
    summaries = {}
    all_generators = [r["generator_id"] for r in inventory()]
    for generator in all_generators:
        metrics_path = OWN / "teacher" / generator / "metrics.json"
        receipt_path = OWN / "teacher" / generator / "TEACHER_RECEIPT.json"
        if not metrics_path.is_file() or not receipt_path.is_file():
            continue
        summaries[generator] = {
            "metrics": json.loads(metrics_path.read_text()),
            "receipt": json.loads(receipt_path.read_text()),
        }
    if set(summaries) != set(all_generators):
        raise RuntimeError(f"Teacher evaluation incomplete: have {sorted(summaries)}, expected {all_generators}")
    summary = {
        "status": "completed",
        "devices": {generator: summaries[generator]["receipt"]["signature"]["device"] for generator in all_generators},
        "benchmark_queries": benchmark_queries,
        "student_pool_policy": "R29 checkpoint own D/E/U/M retrieval; frozen corpus and protocol",
        "teacher_policy": "frozen R19 T0 scores each arm's U and M_exact pools; no qrels in scoring",
        "generators": summaries,
    }
    write_json(R29 / "R29_TEACHER_EVALUATION.json", summary)
    return {"status": "completed", "generators": generators, "summary": str(R29 / "R29_TEACHER_EVALUATION.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index-threads", type=int, default=2)
    parser.add_argument("--benchmark-queries", type=int, default=32)
    args = parser.parse_args()
    generators = args.generator or [r["generator_id"] for r in inventory()]
    print(json.dumps(run(generators, args.device, args.index_threads, args.benchmark_queries), ensure_ascii=False))


if __name__ == "__main__":
    main()

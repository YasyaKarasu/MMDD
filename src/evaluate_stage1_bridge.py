"""Evaluate completed bridge checkpoints with the frozen R26 own-pool protocol."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import evaluate_stage1_r26 as retrieval
import evaluate_stage1_r26_teacher as teacher
from prepare_stage1_r27 import ROOT, read_json, write_json
from run_stage1_bridge import OUT, record


DEST = OUT / "evaluation"
R26 = ROOT / "work/stage1_optimization_r26_20260914"


def prepare_inventory() -> list[dict]:
    source_protocol = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation/PROTOCOL.json"
    source_queries = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation/common/dev_queries.jsonl"
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "common").mkdir(exist_ok=True)
    shutil.copyfile(source_protocol, DEST / "PROTOCOL.json")
    shutil.copyfile(source_queries, DEST / "common/dev_queries.jsonl")
    b0 = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/C2/seed13/checkpoints/step_000178.pt"
    b1 = OUT / "training/B1/seed13/C2/checkpoints/step_000178.pt"
    b1_seed29 = OUT / "training/B1/seed29/C2/checkpoints/step_000178.pt"
    b2 = OUT / "training/B2/seed13/C2/checkpoints/step_000178.pt"
    b2_seed29 = OUT / "training/B2/seed29/C2/checkpoints/step_000178.pt"
    b3 = OUT / "training/B3/seed13/C2/checkpoints/step_000178.pt"
    b3_seed29 = OUT / "training/B3/seed29/C2/checkpoints/step_000178.pt"
    b4 = OUT / "training/B4/seed13/C2/checkpoints/step_000178.pt"
    b4_seed29 = OUT / "training/B4/seed29/C2/checkpoints/step_000178.pt"
    b5_c1_356 = OUT / "training/B5/seed13/C1/checkpoints/step_000356.pt"
    b5_c1_356_seed29 = OUT / "training/B5/seed29/C1/checkpoints/step_000356.pt"
    b5_c1_500 = OUT / "training/B5/seed13/C1/checkpoints/step_000500.pt"
    b5_c1_500_seed29 = OUT / "training/B5/seed29/C1/checkpoints/step_000500.pt"
    b5_c1_659 = OUT / "training/B5/seed13/C1/checkpoints/step_000659.pt"
    b5_c1_659_seed29 = OUT / "training/B5/seed29/C1/checkpoints/step_000659.pt"
    b5 = OUT / "training/B5/seed13/C2/checkpoints/step_000178.pt"
    b5_seed29 = OUT / "training/B5/seed29/C2/checkpoints/step_000178.pt"
    b6 = OUT / "training/B6/seed13/C2/checkpoints/step_000178.pt"
    b6_seed29 = OUT / "training/B6/seed29/C2/checkpoints/step_000178.pt"
    b7 = OUT / "training/B7/seed13/C2/checkpoints/step_000178.pt"
    b7_seed29 = OUT / "training/B7/seed29/C2/checkpoints/step_000178.pt"
    bmodern = R26 / "training/O-NATIVE/seed13/checkpoints/step_000178.pt"
    bmodern_seed29 = R26 / "training/O-NATIVE/seed29/checkpoints/step_000178.pt"
    inventory = [
        {"generator_id": "B0", "seed": 13, "step": 178, "checkpoint": str(b0.resolve())},
        {"generator_id": "B1", "seed": 13, "step": 178, "checkpoint": str(b1.resolve())},
        {"generator_id": "B1_seed29", "seed": 29, "step": 178, "checkpoint": str(b1_seed29.resolve())},
        {"generator_id": "B2", "seed": 13, "step": 178, "checkpoint": str(b2.resolve())},
        {"generator_id": "B2_seed29", "seed": 29, "step": 178, "checkpoint": str(b2_seed29.resolve())},
        {"generator_id": "B3", "seed": 13, "step": 178, "checkpoint": str(b3.resolve())},
        {"generator_id": "B3_seed29", "seed": 29, "step": 178, "checkpoint": str(b3_seed29.resolve())},
        {"generator_id": "B4", "seed": 13, "step": 178, "checkpoint": str(b4.resolve())},
        {"generator_id": "B4_seed29", "seed": 29, "step": 178, "checkpoint": str(b4_seed29.resolve())},
        {"generator_id": "B5_356", "seed": 13, "step": 356, "checkpoint": str(b5_c1_356.resolve()), "training_stage": "C1"},
        {"generator_id": "B5_356_seed29", "seed": 29, "step": 356, "checkpoint": str(b5_c1_356_seed29.resolve()), "training_stage": "C1"},
        {"generator_id": "B5_500", "seed": 13, "step": 500, "checkpoint": str(b5_c1_500.resolve()), "training_stage": "C1"},
        {"generator_id": "B5_500_seed29", "seed": 29, "step": 500, "checkpoint": str(b5_c1_500_seed29.resolve()), "training_stage": "C1"},
        {"generator_id": "B5_659", "seed": 13, "step": 659, "checkpoint": str(b5_c1_659.resolve()), "training_stage": "C1"},
        {"generator_id": "B5_659_seed29", "seed": 29, "step": 659, "checkpoint": str(b5_c1_659_seed29.resolve()), "training_stage": "C1"},
        {"generator_id": "B5", "seed": 13, "step": 178, "checkpoint": str(b5.resolve())},
        {"generator_id": "B5_seed29", "seed": 29, "step": 178, "checkpoint": str(b5_seed29.resolve())},
        {"generator_id": "B6", "seed": 13, "step": 178, "checkpoint": str(b6.resolve())},
        {"generator_id": "B6_seed29", "seed": 29, "step": 178, "checkpoint": str(b6_seed29.resolve())},
        {"generator_id": "B7", "seed": 13, "step": 178, "checkpoint": str(b7.resolve())},
        {"generator_id": "B7_seed29", "seed": 29, "step": 178, "checkpoint": str(b7_seed29.resolve())},
        {"generator_id": "B-modern", "seed": 13, "step": 178, "checkpoint": str(bmodern.resolve()),
         "training_stage": "R26 O-NATIVE", "external_reference": "R25 C1 step659 + R26 O-NATIVE C2 step178"},
        {"generator_id": "B-modern_seed29", "seed": 29, "step": 178, "checkpoint": str(bmodern_seed29.resolve()),
         "training_stage": "R26 O-NATIVE", "external_reference": "R25 C1 step659 + R26 O-NATIVE C2 step178"},
    ]
    write_json(DEST / "MODEL_INVENTORY.json", inventory)
    return inventory


def run_own(generators: list[str], device: str) -> dict:
    inventory = prepare_inventory()
    retrieval.OUT = DEST
    completed = []
    for generator in generators:
        result = retrieval.evaluate(generator, device, index_threads=2)
        write_json(DEST / "node_receipts" / f"{generator}.json", {"status": "completed", "result": result, "protocol": record(DEST / "PROTOCOL.json")})
        completed.append(generator)
    return {"completed": completed, "inventory": inventory}


def run_teacher(generators: list[str], device: str, benchmark_queries: int = 32) -> dict:
    prepare_inventory()
    teacher.OUT = DEST
    return teacher.run(generators, device, benchmark_queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", action="append", choices=(
        "B0", "B1", "B1_seed29", "B2", "B2_seed29", "B3", "B3_seed29",
        "B4", "B4_seed29", "B5_356", "B5_356_seed29", "B5_500", "B5_500_seed29", "B5_659", "B5_659_seed29",
        "B5", "B5_seed29", "B6", "B6_seed29", "B7", "B7_seed29", "B-modern", "B-modern_seed29"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--benchmark-queries", type=int, default=32)
    args = parser.parse_args()
    generators = args.generator or ["B0", "B1"]
    result = run_teacher(generators, args.device, args.benchmark_queries) if args.teacher else run_own(generators, args.device)
    print(json.dumps(result, ensure_ascii=False))

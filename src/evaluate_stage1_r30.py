"""Run R30 own-pool retrieval and the frozen T0 reranker.

The numerical retrieval implementation is the audited R26/Bridge evaluator;
this module only supplies an R30-local protocol, query population, and model
inventory so historical output directories remain read-only.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import evaluate_stage1_r26 as retrieval
import evaluate_stage1_r26_teacher as teacher
from prepare_stage1_r27 import ROOT
from run_stage1_bridge import sha256, write_json


OUT = ROOT / "work/stage1_r30_c1_et_20260916"
BRIDGE_EVAL = ROOT / "work/stage1_bridge_20260915/evaluation"


def inventory() -> list[dict[str, Any]]:
    c1 = OUT / "C1/F-P"
    bridge = ROOT / "work/stage1_bridge_20260915/training/B5"
    values = [
        {"generator_id": "STOP356_s13", "seed": 13, "step": 356, "checkpoint": str((bridge / "seed13/C1/checkpoints/step_000356.pt").resolve()), "source": "Bridge B5 existing control"},
        {"generator_id": "STOP356_s29", "seed": 29, "step": 356, "checkpoint": str((bridge / "seed29/C1/checkpoints/step_000356.pt").resolve()), "source": "Bridge B5 existing control"},
        {"generator_id": "JOINT659_s13", "seed": 13, "step": 659, "checkpoint": str((bridge / "seed13/C1/checkpoints/step_000659.pt").resolve()), "source": "Bridge B5 existing control"},
        {"generator_id": "JOINT659_s29", "seed": 29, "step": 659, "checkpoint": str((bridge / "seed29/C1/checkpoints/step_000659.pt").resolve()), "source": "Bridge B5 existing control"},
    ]
    for seed in (13, 29):
        for step in (500, 659):
            values.append({
                "generator_id": f"F-P{step}_s{seed}", "seed": seed, "step": step,
                "checkpoint": str((c1 / f"seed{seed}/checkpoints/step_{step:06d}.pt").resolve()),
                "training_stage": "R30 C1 F-P",
            })
    etnat = OUT / "C1/F-P-ETNAT"
    for seed in (13, 29):
        for step in (500, 659):
            checkpoint = etnat / f"seed{seed}/checkpoints/step_{step:06d}.pt"
            if checkpoint.is_file():
                values.append({
                    "generator_id": f"F-P-ETNAT{step}_s{seed}", "seed": seed, "step": step,
                    "checkpoint": str(checkpoint.resolve()),
                    "training_stage": "R30 C1 F-P-ETNAT",
                })
    selection_path = OUT / "C1_SELECTION.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        recipe = selection.get("recipe")
        if selection.get("status") == "selected" and recipe in {"F-P", "F-P-ETNAT"}:
            c2 = OUT / f"C2-CHECK/{recipe}"
            for seed in (13, 29):
                for step in (89, 178):
                    checkpoint = c2 / f"seed{seed}/checkpoints/step_{step:06d}.pt"
                    if checkpoint.is_file():
                        values.append({
                            "generator_id": f"C2-{recipe}{step}_s{seed}", "seed": seed, "step": step,
                            "checkpoint": str(checkpoint.resolve()),
                            "training_stage": f"R30 C2-CHECK {recipe}",
                        })
    return values


def prepare() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "common").mkdir(exist_ok=True)
    protocol_source = BRIDGE_EVAL / "PROTOCOL.json"
    queries_source = BRIDGE_EVAL / "common/dev_queries.jsonl"
    shutil.copyfile(protocol_source, OUT / "PROTOCOL.json")
    shutil.copyfile(queries_source, OUT / "common/dev_queries.jsonl")
    models = inventory()
    missing = [row["checkpoint"] for row in models if not Path(row["checkpoint"]).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    write_json(OUT / "MODEL_INVENTORY.json", models)
    receipt = {
        "status": "complete",
        "protocol": {"path": str(protocol_source.resolve()), "sha256": sha256(protocol_source)},
        "queries": {"path": str(queries_source.resolve()), "sha256": sha256(queries_source), "rows": sum(1 for line in queries_source.open() if line.strip())},
        "models": [{**row, "sha256": sha256(Path(row["checkpoint"]))} for row in models],
        "retrieval_implementation": {"path": str((ROOT / "src/evaluate_stage1_r26.py").resolve()), "sha256": sha256(ROOT / "src/evaluate_stage1_r26.py")},
        "teacher_implementation": {"path": str((ROOT / "src/evaluate_stage1_r26_teacher.py").resolve()), "sha256": sha256(ROOT / "src/evaluate_stage1_r26_teacher.py")},
        "policy": "own Student ANN/exact D/E/U/M; Equal C100; fixed historical T0; no dev labels in retrieval",
    }
    write_json(OUT / "MODEL_AND_INDEX_LOCK.json", receipt)
    return receipt


def run_own(generators: list[str], device: str, index_threads: int) -> dict[str, Any]:
    prepare()
    retrieval.OUT = OUT
    retrieval.PRELOAD_ALL = True
    completed = []
    for generator in generators:
        completed.append(retrieval.evaluate(generator, device, index_threads=index_threads, legacy_diagnostics=False))
    return {"completed": completed}


def run_teacher(generators: list[str], device: str, benchmark_queries: int) -> dict[str, Any]:
    prepare()
    teacher.OUT = OUT
    return teacher.run(generators, device, benchmark_queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generator", action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index-threads", type=int, default=2)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--benchmark-queries", type=int, default=0)
    args = parser.parse_args()
    if args.prepare and not args.generator:
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
        return
    generators = args.generator or [row["generator_id"] for row in inventory() if row["generator_id"].startswith("F-P")]
    result = run_teacher(generators, args.device, args.benchmark_queries) if args.teacher else run_own(generators, args.device, args.index_threads)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

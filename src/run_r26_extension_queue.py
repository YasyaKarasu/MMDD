"""Sequential conditional C2 jobs with own evaluations and downstream followups."""
from __future__ import annotations

import json
import time

from prepare_stage1_r26 import ROOT,OUT,file_record
from run_r26_followups import execute,await_artifact
from run_stage1_r25 import _json
from train_stage1_r26_extension import ARMS


def run() -> None:
    await_artifact(OUT / "EVALUATION_QUEUE_RESULT.json")
    # Image reader bursts can occupy nearly all GPU1 memory. Finish that
    # workload before the remaining extension trainers.
    await_artifact(OUT / "stage2/pilot/B13/PILOT_RECEIPT.json")
    results = []
    for arm in ARMS:
        for seed in (13,29):
            execute("train_stage1_r26_extension.py",["--arm",arm,"--seed",str(seed),"--device","cuda:1"],f"extension_{arm}_{seed}.log")
            receipt = OUT / "training" / arm / f"seed{seed}" / "C2_COMPLETION_RECEIPT.json"
            results.append(file_record(receipt))
    inventory_path = OUT / "MODEL_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    existing = {r["generator_id"] for r in inventory}
    names = []
    for arm in ARMS:
        for seed in (13,29):
            for step in (0,89,178):
                name = f"R26-{arm}/seed{seed}/step{step}"
                path = OUT / "training" / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
                if name not in existing:
                    inventory.append({"generator_id":name,"seed":seed,"step":step,"checkpoint":str(path),**file_record(path)})
                if step:
                    names.append(name)
    _json(inventory_path,inventory)
    _json(OUT / "EXTENSION_TRAINING_RECEIPT.json",{"execution_status":"ran","jobs":results,"generators":names})
    # CPU postprocessing and Teacher queues continue independently. Keep one
    # own-lake evaluation on GPU0, whose headroom was verified by B13/pre-C1.
    for name in names:
        execute("evaluate_stage1_r26.py",["--generator",name,"--device","cuda:0"],"extension_eval_"+name.replace("/","_")+".log")
    _json(OUT / "EXTENSION_EVALUATION_RECEIPT.json",{"execution_status":"ran","generators":names})
    execute("analyze_stage1_r26.py",[],"statistics_with_extension.log")


if __name__ == "__main__":
    run()

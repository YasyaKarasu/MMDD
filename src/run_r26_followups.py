"""Serialize the remaining large jobs after their currently running prerequisites."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from prepare_stage1_r26 import ROOT,OUT


def await_artifact(path: Path) -> None:
    print(json.dumps({"waiting_for":str(path)}),flush=True)
    while not path.exists():
        time.sleep(20)


def execute(script: str, args: list[str], log_name: str) -> None:
    print(json.dumps({"start":script,"args":args}),flush=True)
    with (OUT / "logs" / log_name).open("w") as handle:
        result = subprocess.run([sys.executable,str(ROOT / "src" / script),*args],cwd="/tmp",stdout=handle,stderr=subprocess.STDOUT)
    print(json.dumps({"exit":script,"code":result.returncode,"log":log_name}),flush=True)
    if result.returncode:
        raise RuntimeError(f"{script} failed; see {log_name}")


def run(lane: str) -> None:
    if lane == "teacher":
        await_artifact(OUT / "teacher/Qwen-Raw/TEACHER_RECEIPT.json")
        execute("evaluate_stage1_r26_teacher.py",["--generator","B13","--device","cpu"],"teacher_B13_cpu.log")
        execute("evaluate_r26_b13_teacher_fusion.py",["--without-column"],"B13_teacher_then_fusion_interim.log")
        await_artifact(OUT / "EVALUATION_QUEUE_RESULT.json")
        execute("evaluate_stage1_r26_teacher.py",["--device","cpu"],"teacher_all_cpu.log")
        await_artifact(OUT / "fusion/columns/COLUMN_RECEIPT.json")
        execute("evaluate_r26_b13_teacher_fusion.py",[],"B13_teacher_then_fusion_complete.log")
    elif lane == "generation":
        await_artifact(OUT / "stage2/pilot/Qwen-Raw/PILOT_RECEIPT.json")
        execute("build_stage1_r26_columns.py",["--device","cuda:1"],"column_supplement.log")
        execute("run_stage2_r26.py",["--generator","B13"],"stage2_pilot_B13.log")
    elif lane == "column":
        await_artifact(OUT / "fusion/columns/COLUMN_RECEIPT.json")
        await_artifact(OUT / "EVALUATION_QUEUE_RESULT.json")
        execute("evaluate_stage1_r26_column.py",["--device","cpu"],"column_fusion.log")
    elif lane == "cells":
        for generator in ("Qwen-Raw","B13"):
            await_artifact(OUT / "stage2/pilot" / generator / "PILOT_RECEIPT.json")
            execute("audit_stage2_r26_cells.py",["--generator",generator],"cell_audit_"+generator+".log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane",choices=("teacher","generation","column","cells"),required=True)
    run(parser.parse_args().lane)

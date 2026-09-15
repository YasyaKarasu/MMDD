"""Track real R28 processes and finalize only after every planned evaluation exists."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_stage1_r27 import write_json
from prepare_stage1_r28 import ROOT, OUT, FAMILIES


def alive(pid: int) -> bool:
    try:
        os.kill(pid,0)
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except (ProcessLookupError, FileNotFoundError):
        return False
    return state != "Z"


def run(t0_pid: int) -> None:
    while True:
        training = []
        for arm in FAMILIES:
            kind = "student" if arm.startswith("S-") else "teacher"
            for seed in (13,29):
                path = OUT / kind / arm / f"seed{seed}/EXECUTION.json"
                r = json.loads(path.read_text())
                training.append({"arm":arm,"seed":seed,"status":r["status"],"updates":r["updates"],
                    "pid":r["pid"],"process_alive":alive(r["pid"]) if r["status"] == "running" else False,
                    "epochs":[c["epoch"] for c in r["checkpoints"]]})
        queues = []
        for p in sorted(OUT.glob("EVALUATION_QUEUE_seed*.json")):
            r = json.loads(p.read_text())
            queues.append({"seed":r["seed"],"ledger":str(p),"expected":len(r["jobs"]),
                           "worker_pid":r["worker_pid"],"process_alive":alive(r["worker_pid"]),
                           "counts":{s:sum(j["status"] == s for j in r["jobs"]) for s in {j["status"] for j in r["jobs"]}}})
        teacher_count = len(list((OUT / "teacher/evaluation").glob("**/EVALUATION_RECEIPT.json")))
        own_count = len(list((OUT / "student/own/rankings").glob("**/R28_EVALUATION_RECEIPT.json")))
        t0_done = (OUT / "teacher/evaluation/T0/EVALUATION_RECEIPT.json").exists()
        status = {"monitor_pid":os.getpid(),"training":training,"evaluation_queues":queues,
                  "teacher_evaluations":teacher_count,"teacher_expected":31,"student_evaluations":own_count,"student_expected":36,
                  "T0":{"receipt_exists":t0_done,"pid":t0_pid,"process_alive":alive(t0_pid) if not t0_done else False},
                  "status":"running","updated_unix":time.time()}
        write_json(OUT / "CURRENT_STATUS.json",status)
        if teacher_count == 31 and own_count == 36 and all(j["status"] == "completed" for j in training):
            break
        if any(j["status"] == "failed" or (j["status"] == "running" and not j["process_alive"]) for j in training):
            status["status"] = "needs_inspection_training"
            write_json(OUT / "CURRENT_STATUS.json",status)
            return
        if (not t0_done and not alive(t0_pid)) or any(not q["process_alive"] and q["counts"].get("completed",0) != q["expected"] for q in queues):
            status["status"] = "needs_inspection_evaluation"
            write_json(OUT / "CURRENT_STATUS.json",status)
            return
        time.sleep(20)
    for name,extra in (("analyze_stage1_r28.py",["--bootstrap"]),("plot_stage1_r28.py",[]),("finalize_stage1_r28.py",[])):
        log_path = OUT / "logs" / f"final-{name}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen([sys.executable,str(ROOT/"src"/name),*extra],cwd="/tmp/mmdd-r28-checks",stdout=log,stderr=subprocess.STDOUT)
            write_json(OUT / "FINALIZATION_STATUS.json",{"status":"running","stage":name,"pid":process.pid,"log":str(log_path)})
            code = process.wait()
        if code:
            write_json(OUT / "FINALIZATION_STATUS.json",{"status":"failed","stage":name,"returncode":code,"log":str(log_path)})
            return
    write_json(OUT / "FINALIZATION_STATUS.json",{"status":"completed_pending_agent_completion_audit"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0-pid",required=True,type=int)
    args = parser.parse_args()
    run(args.t0_pid)

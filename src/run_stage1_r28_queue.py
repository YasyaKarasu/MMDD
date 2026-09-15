"""Run the fixed 12-job matrix: three Teachers and one Student per GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_stage1_r27 import write_json
from prepare_stage1_r28 import ROOT, OUT, FAMILIES


def run() -> None:
    assert json.loads((OUT / "R28_OBJECTIVE_AUDIT.json").read_text())["status"] == "pass"
    isolated = Path("/tmp/mmdd-r28-runtime")
    isolated.mkdir(exist_ok=True)
    jobs, handles = [], {}
    log_dir = OUT / "logs"
    log_dir.mkdir(exist_ok=True)
    for gpu, seed in enumerate((13,29)):
        for arm in FAMILIES:
            jobs.append({"arm":arm, "seed":seed, "gpu":gpu, "status":"pending"})
    def ledger():
        write_json(OUT / "EXECUTION_LEDGER.json", {"status":"running" if any(j["status"] in ("running","pending") for j in jobs) else "training_terminal_evaluation_pending",
            "G0":"pass", "G1":"pass", "queue_pid":os.getpid(), "jobs":jobs,
            "evaluation_status":"pending", "stage2_jobs":0, "matrix_limit":12})
    while any(j["status"] in ("running","pending") for j in jobs):
        for i,(process, log) in list(handles.items()):
            code = process.poll()
            if code is not None:
                jobs[i].update({"status":"completed" if code == 0 else "failed", "returncode":code})
                log.close()
                del handles[i]
        for i,job in enumerate(jobs):
            if job["status"] != "pending":
                continue
            student = job["arm"].startswith("S-")
            active = [j for j in jobs if j["gpu"] == job["gpu"] and j["status"] == "running"]
            if student and any(j["arm"].startswith("S-") for j in active):
                continue
            kind = "student" if student else "teacher"
            existing = OUT / kind / job["arm"] / f"seed{job['seed']}/EXECUTION.json"
            if existing.exists():
                raise FileExistsError("Existing training job requires explicit inspection; queue never restarts")
            command = [sys.executable, str(ROOT/"src/train_stage1_r28.py"), "--arm", job["arm"],
                       "--seed", str(job["seed"]), "--device", f"cuda:{job['gpu']}"]
            log = (log_dir/f"{job['arm']}-seed{job['seed']}.log").open("w")
            process = subprocess.Popen(command, cwd=isolated, stdout=log, stderr=subprocess.STDOUT)
            handles[i] = (process,log)
            job.update({"status":"running", "pid":process.pid, "command":command})
        ledger()
        if handles:
            time.sleep(10)
    ledger()


if __name__ == "__main__":
    run()

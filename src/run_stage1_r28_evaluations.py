"""Consume registered checkpoints, optionally splitting Student and Teacher queues."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_stage1_r27 import write_json
from prepare_stage1_r28 import ROOT, OUT
from evaluate_stage1_r28_student import OWN


def alive(pid: int) -> bool:
    try:
        os.kill(pid,0)
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except (ProcessLookupError, FileNotFoundError):
        return False
    return state != "Z"


def jobs_for(seed: int, device: str, kind: str = "both") -> list[dict]:
    jobs = []
    for spec in json.loads((OWN / "MODEL_INVENTORY.json").read_text()):
        if spec["seed"] != seed:
            continue
        gid = spec["generator_id"]
        jobs.append({"kind":"student","id":gid,"checkpoint":spec["checkpoint"],
            "receipt":str(OWN / "rankings" / gid / "R28_EVALUATION_RECEIPT.json"),
            "external_running":str(OWN / "rankings" / gid / "RUNNING.json"),
            "training":str(OUT / "student" / spec["arm"] / f"seed{seed}/EXECUTION.json"),
            "command":[sys.executable,str(ROOT/"src/evaluate_stage1_r28_student.py"),"--generator",gid,"--device",device],
            "status":"pending"})
    for arm in ("T-EDGE-CONT","T-PATH-SPLIT-LSE","T-PATH-SPLIT-COV"):
        for epoch in (.5,1,2,3,5):
            gid = f"{arm}/seed{seed}/epoch{epoch:g}"
            jobs.append({"kind":"teacher","id":gid,
                "checkpoint":str(OUT / "teacher" / arm / f"seed{seed}/checkpoints/step_{int(epoch*1424):06d}.pt"),
                "receipt":str(OUT / "teacher/evaluation" / gid / "EVALUATION_RECEIPT.json"),
                "training":str(OUT / "teacher" / arm / f"seed{seed}/EXECUTION.json"),
                "command":[sys.executable,str(ROOT/"src/evaluate_stage1_r28_teacher_fast.py"),"--arm",arm,"--seed",str(seed),
                           "--epoch",str(epoch),"--device",device],"status":"pending"})
    return [j for j in jobs if kind == "both" or j["kind"] == kind]


def run(seed: int, device: str, kind: str = "both", *,
        wait_pid: int | None = None, wait_receipt: Path | None = None) -> None:
    jobs = jobs_for(seed, device, kind)
    suffix = "" if kind == "both" else f"_{kind}"
    ledger = OUT / f"EVALUATION_QUEUE_seed{seed}{suffix}.json"
    if wait_pid is not None:
        write_json(ledger,{"worker_pid":os.getpid(),"seed":seed,"device":device,"kind":kind,
            "status":"waiting_for_inherited_job","inherited_pid":wait_pid,"jobs":jobs})
        # A paused old supervisor can leave a finished child as a zombie.
        # Wait for the child itself to finish before starting another same-kind job.
        while alive(wait_pid):
            time.sleep(10)
        assert wait_receipt is not None and wait_receipt.is_file(), "Inherited evaluation ended without a receipt"
        assert json.loads(wait_receipt.read_text())["status"] == "completed"
    # Round robin keeps Teacher trajectories progressing while Student own indices build.
    prefer = "student"
    while any(j["status"] == "pending" for j in jobs):
        available = []
        for job in jobs:
            if job["status"] != "pending":
                continue
            if Path(job["receipt"]).exists():
                job["status"] = "completed"
                continue
            external = Path(job["external_running"]) if "external_running" in job else None
            if external and external.exists():
                rec = json.loads(external.read_text())
                if alive(rec["pid"]):
                    continue
                job["status"] = "external_terminal_missing_receipt"
                continue
            if Path(job["checkpoint"]).with_suffix(".json").exists():
                available.append(job)
            else:
                train_path = Path(job["training"])
                if train_path.exists():
                    training = json.loads(train_path.read_text())
                    if training["status"] == "failed":
                        job["status"] = "missing_checkpoint_training_failed"
        write_json(ledger,{"worker_pid":os.getpid(),"seed":seed,"device":device,"jobs":jobs})
        if not available:
            if any(j["status"] == "pending" for j in jobs):
                time.sleep(20)
            continue
        job = next((j for j in available if j["kind"] == prefer),available[0])
        prefer = "teacher" if job["kind"] == "student" else "student"
        log_path = OUT / "logs" / ("eval-" + job["id"].replace("/","-") + ".log")
        with log_path.open("w") as log:
            process = subprocess.Popen(job["command"],cwd="/tmp/mmdd-r28-checks",stdout=log,stderr=subprocess.STDOUT)
            job.update({"status":"running","pid":process.pid,"log":str(log_path)})
            write_json(ledger,{"worker_pid":os.getpid(),"seed":seed,"device":device,"jobs":jobs})
            code = process.wait()
        job.update({"status":"completed" if code == 0 and Path(job["receipt"]).exists() else "failed","returncode":code})
    write_json(ledger,{"worker_pid":os.getpid(),"seed":seed,"device":device,"status":"terminal","jobs":jobs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,required=True,choices=(13,29))
    parser.add_argument("--device",required=True)
    parser.add_argument("--kind",choices=("both","student","teacher"),default="both")
    parser.add_argument("--wait-pid",type=int)
    parser.add_argument("--wait-receipt",type=Path)
    parser.add_argument("--preview",action="store_true")
    args = parser.parse_args()
    if (args.wait_pid is None) != (args.wait_receipt is None) or (args.wait_pid is not None and args.wait_pid <= 0):
        parser.error("--wait-pid must be positive and supplied together with --wait-receipt")
    if args.preview:
        print(json.dumps(jobs_for(args.seed,args.device,args.kind)))
    else:
        run(args.seed,args.device,args.kind,wait_pid=args.wait_pid,wait_receipt=args.wait_receipt)

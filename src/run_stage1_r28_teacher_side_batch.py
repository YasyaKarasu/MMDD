"""Run the four registered pending Path Teacher trajectories alongside Student evals."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from prepare_stage1_r27 import record, write_json
from prepare_stage1_r28 import OUT
from run_stage1_r28_evaluations import alive, jobs_for


def main() -> None:
    receipt = OUT / "TEACHER_PARALLEL_BATCH.json"
    assert not receipt.exists(), "Inspect the existing batch before restarting"
    queues = {seed: json.loads((OUT / f"EVALUATION_QUEUE_seed{seed}_teacher.json").read_text())
              for seed in (13, 29)}
    paused = []
    state = {"status": "preparing", "pid": os.getpid(), "code": record(Path(__file__)),
             "jobs": [], "supervisors": {seed: q["worker_pid"] for seed, q in queues.items()},
             "scientific_nodes_added": 0, "training_jobs_restarted": 0}
    lock = threading.Lock()

    def save() -> None:
        write_json(receipt, state)

    try:
        # Stop dispatch only. Existing Edge children continue to completion.
        for seed, queue in queues.items():
            pid = queue["worker_pid"]
            assert b"run_stage1_r28_evaluations.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)
            while Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] != "T":
                time.sleep(.01)
            snapshot = json.loads((OUT / f"EVALUATION_QUEUE_seed{seed}_teacher.json").read_text())
            assert snapshot["worker_pid"] == pid
            assert all(j["id"].startswith("T-EDGE-CONT/") for j in snapshot["jobs"] if j["status"] == "running")
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            for child in children:
                command = Path(f"/proc/{child}/cmdline").read_bytes()
                assert b"T-EDGE-CONT" in command or not alive(int(child))
        trajectories = []
        for seed, device in ((13, "cuda:0"), (29, "cuda:1")):
            registered = jobs_for(seed, device, "teacher")
            original = {j["id"]: j["command"] for j in queues[seed]["jobs"]}
            for arm in ("T-PATH-SPLIT-LSE", "T-PATH-SPLIT-COV"):
                trajectory = [j for j in registered if j["id"].startswith(arm + "/")
                              and not Path(j["receipt"]).exists()]
                for job in trajectory:
                    assert job["command"] == original[job["id"]]
                    assert Path(job["checkpoint"]).with_suffix(".json").is_file()
                trajectories.append(trajectory)
                state["jobs"].extend(trajectory)
        state.update(status="running", started_unix=time.time())
        save()

        def run_trajectory(jobs: list[dict]) -> None:
            for job in jobs:
                log_path = OUT / "logs" / ("eval-" + job["id"].replace("/", "-") + ".log")
                assert not Path(job["receipt"]).exists()
                with log_path.open("w") as log:
                    process = subprocess.Popen(job["command"], cwd="/tmp/mmdd-r28-checks",
                                               stdout=log, stderr=subprocess.STDOUT)
                    with lock:
                        job.update(status="running", pid=process.pid, log=str(log_path), started_unix=time.time())
                        save()
                    code = process.wait()
                ok = code == 0 and Path(job["receipt"]).exists()
                if ok:
                    ok = json.loads(Path(job["receipt"]).read_text())["status"] == "completed"
                with lock:
                    job.update(status="completed" if ok else "failed", returncode=code, finished_unix=time.time())
                    save()
                if not ok:
                    raise RuntimeError(f"Registered evaluation failed: {job['id']}; inspect {log_path}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_trajectory, jobs) for jobs in trajectories]
            for future in futures:
                future.result()
        state.update(status="completed", finished_unix=time.time())
    except BaseException:
        state["status"] = "needs_inspection"
        raise
    finally:
        for pid in paused:
            if alive(pid):
                os.kill(pid, signal.SIGCONT)
        state["supervisors_resumed"] = paused
        save()


if __name__ == "__main__":
    main()

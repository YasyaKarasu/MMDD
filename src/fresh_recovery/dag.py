"""EXECUTION_DAG scheduler: independent stages as subprocesses on assigned GPUs.

Completion is detected from each stage's own ``POST_RUN.json`` / report file
(never from a plan), so a re-run resumes at the first unfinished node.  A
failed stage is retried at most once with the identical configuration and the
failure is recorded in ``ERROR_LEDGER.jsonl``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Paths
from .io import write_json

PYTHON = sys.executable
ENTRY = Path(__file__).resolve().parent.parent / "run_fresh_recovery.py"


@dataclass
class Node:
    name: str
    args: list[str]
    parents: list[str]
    gpu: bool
    done: object  # callable(paths, seed) -> bool
    kind: str = "prepare"
    attempts: int = 0
    process: subprocess.Popen | None = None
    gpu_index: int | None = None
    started: float = 0.0


def _exists(rel: str):
    def check(paths: Paths, seed: int) -> bool:
        return (paths.work_dir / rel.format(seed=seed)).exists()
    return check


def _post_run(stage: str):
    def check(paths: Paths, seed: int) -> bool:
        path = paths.work_dir / f"seed{seed}" / stage / "POST_RUN.json"
        return path.exists() and json.loads(path.read_text()).get("status") == "COMPLETE"
    return check


def _selection(pair: str, phase: str):
    def check(paths: Paths, seed: int) -> bool:
        path = paths.work_dir / f"seed{seed}" / f"SELECTION_{pair.upper()}_{phase}.json"
        return path.exists() and json.loads(path.read_text()).get("status") == "SELECTED"
    return check


def build_nodes() -> list[Node]:
    n = []
    n.append(Node("ROWS", ["build-rows"], [], False, _exists("rows/index.json")))
    for split in ("train", "dev"):
        n.append(Node(f"RAW_{split}", ["build-raw", "--split", split], ["ROWS"], True, _exists(f"raw/{split}/RAW_REPORT.json")))
    n.append(Node("LISTS", ["build-lists", "--seed", "{seed}"], ["RAW_train"], False, _exists("lists/seed{seed}/L0_REPORT.json")))
    n.append(Node("T_INIT", ["init-teacher", "--seed", "{seed}"], [], False, _exists("seed{seed}/T_INIT/INIT_LINEAGE.json"), "initialization"))
    n.append(Node("INIT_REFERENCE", ["eval-init", "--seed", "{seed}"], ["RAW_dev", "ROWS"], True, _exists("seed{seed}/INIT_REFERENCE/INIT_REFERENCE.json")))
    n.append(Node("VERIFY", ["verify-integration", "--seed", "{seed}"], ["RAW_train", "RAW_dev", "LISTS", "T_INIT"], True,
                  _exists("seed{seed}/VERIFY_INTEGRATION/VERIFY_INTEGRATION.json")))
    n.append(Node("T_BOOT", ["train-bootstrap", "--seed", "{seed}"], ["LISTS", "T_INIT", "RAW_dev", "VERIFY"], True, _post_run("T_BOOT"), "train"))
    n.append(Node("L1", ["refresh-lists", "--seed", "{seed}"], ["T_BOOT"], True, _post_run("L1")))
    for arm in ("S_SUP_NATIVE", "S_KD_NATIVE", "S_QT_SUP", "S_QT_KD"):
        n.append(Node(f"{arm}_C1", ["train-c1", "--seed", "{seed}", "--arm", arm], ["L1", "INIT_REFERENCE"], True, _post_run(f"{arm}_C1"), "train"))
    for pair, arms in (("native", ("S_SUP_NATIVE", "S_KD_NATIVE")), ("qt", ("S_QT_SUP", "S_QT_KD"))):
        n.append(Node(f"SELECT_{pair}_C1", ["select-c1", "--seed", "{seed}", "--pair", pair], [f"{a}_C1" for a in arms], False, _selection(pair, "C1")))
        n.append(Node(f"{pair.upper()}_C1_SELECTION_GRAPH", ["build-c2-graphs", "--seed", "{seed}", "--pair", pair], [f"SELECT_{pair}_C1"], True,
                      _post_run(f"{pair.upper()}_C1_SELECTION_GRAPH")))
        for arm in arms:
            n.append(Node(f"{arm}_C2", ["train-c2", "--seed", "{seed}", "--arm", arm], [f"{pair.upper()}_C1_SELECTION_GRAPH"], True, _post_run(f"{arm}_C2"), "train"))
        n.append(Node(f"SELECT_{pair}_C2", ["select-c2", "--seed", "{seed}", "--pair", pair], [f"{a}_C2" for a in arms], False, _selection(pair, "C2")))
    n.append(Node("HARD32", ["mine-qt-hard", "--seed", "{seed}"], ["SELECT_native_C2"], True, _post_run("HARD32")))
    n.append(Node("T_QT", ["train-qt-teacher", "--seed", "{seed}"], ["HARD32", "T_INIT"], True, _post_run("T_QT"), "train"))
    n.append(Node("T_QT_FROM_BOOT", ["train-qt-teacher-from-boot", "--seed", "{seed}"], ["HARD32", "T_BOOT"], True,
                  _post_run("T_QT_FROM_BOOT"), "train"))
    n.append(Node("T_QT_CONT", ["train-qt-cont", "--seed", "{seed}"], ["T_QT", "SELECT_native_C2"], True, _post_run("T_QT_CONT"), "train"))
    n.append(Node("T_PATH", ["train-path-teacher", "--seed", "{seed}"], ["T_QT", "SELECT_native_C2"], True, _post_run("T_PATH"), "train"))
    n.append(Node("EVAL_DEV", ["evaluate-dev", "--seed", "{seed}"], ["T_QT", "T_QT_FROM_BOOT", "T_QT_CONT", "T_PATH", "SELECT_native_C2", "SELECT_qt_C2", "RAW_dev"], True,
                  _exists("seed{seed}/EVAL_DEV/RESULTS.json"), "evaluation"))
    n.append(Node("LATENCY", ["measure-latency", "--seed", "{seed}"], ["EVAL_DEV"], True, _exists("seed{seed}/EVAL_DEV/LATENCY.json")))
    n.append(Node("FREEZE_MODELS", ["freeze-models", "--seed", "{seed}"], ["EVAL_DEV", "LATENCY", "T_QT_FROM_BOOT", "T_QT_CONT", "T_PATH"], False,
                  _exists("seed{seed}/MODEL_LOCK.json"), "freeze"))
    # Existing test data/GT are intentionally not opened until this seed's
    # complete model/config lock exists.  This prevents a pre-freeze test
    # read from influencing selection or budget decisions.
    n.append(Node("RAW_test", ["build-raw", "--split", "test"], ["FREEZE_MODELS"], True,
                  _exists("raw/test/RAW_REPORT.json")))
    n.append(Node("EVAL_TEST", ["evaluate-test", "--seed", "{seed}"], ["FREEZE_MODELS", "RAW_test"], True,
                  _exists("seed{seed}/EVAL_TEST/RESULTS.json"), "evaluation"))
    return n


def run_dag(
    paths: Paths,
    *,
    seed: int,
    gpus: list[int],
    max_stage: str | None = None,
    poll: float = 20.0,
    gpu_slots_per_device: int = 1,
) -> dict:
    if gpu_slots_per_device not in (1, 2):
        raise ValueError("gpu_slots_per_device must be 1 or 2")
    nodes = {node.name: node for node in build_nodes()}
    if max_stage is not None:
        keep = set()

        def walk(name):
            if name in keep:
                return
            keep.add(name)
            for p in nodes[name].parents:
                walk(p)
        walk(max_stage)
        nodes = {k: v for k, v in nodes.items() if k in keep}
    # A repeated physical index represents one bounded co-resident slot.  The
    # experiment protocol permits no more than two jobs on one GPU.
    free_gpus = [gpu for gpu in gpus for _ in range(gpu_slots_per_device)]
    ledger = paths.work_dir / f"seed{seed}" / "ERROR_LEDGER.jsonl"
    events = paths.work_dir / f"seed{seed}" / "SCHEDULE_EVENTS.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    log_dir = paths.work_dir / f"seed{seed}" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cpu_running = 0

    def event(payload: dict) -> None:
        payload = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}
        with events.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        print(json.dumps(payload), flush=True)

    status = {name: ("DONE" if node.done(paths, seed) else "PENDING") for name, node in nodes.items()}
    failed: set[str] = set()
    while any(s in ("PENDING", "RUNNING") for s in status.values()):
        # launch ready nodes
        for name, node in nodes.items():
            if status[name] != "PENDING":
                continue
            if any(status.get(p) != "DONE" for p in node.parents):
                if any(status.get(p) in ("FAILED", "BLOCKED") for p in node.parents):
                    status[name] = "BLOCKED"
                    event({"node": name, "status": "BLOCKED", "reason": "parent failed"})
                continue
            if node.gpu:
                if not free_gpus:
                    continue
                node.gpu_index = free_gpus.pop(0)
            elif cpu_running >= 2:
                continue
            args = [PYTHON, str(ENTRY), "--work-dir", str(paths.work_dir), "--dataset-root", str(paths.dataset_root),
                    "--backbone-dir", str(paths.backbone_dir), "--package-dir", str(paths.package_dir)]
            args += [a.format(seed=seed) for a in node.args]
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = "8"
            if node.gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(node.gpu_index)
                args += ["--device", "cuda:0"]
            else:
                cpu_running += 1
            node.attempts += 1
            log_path = log_dir / f"{name}.attempt{node.attempts}.log"
            node.process = subprocess.Popen(args, env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT, cwd=str(ENTRY.parent.parent))
            node.started = time.time()
            status[name] = "RUNNING"
            event({"node": name, "status": "STARTED", "attempt": node.attempts, "gpu": node.gpu_index, "pid": node.process.pid,
                   "log": str(log_path)})
        # reap finished
        for name, node in nodes.items():
            if status[name] != "RUNNING" or node.process is None or node.process.poll() is None:
                continue
            rc = node.process.returncode
            elapsed = time.time() - node.started
            if node.gpu:
                free_gpus.append(node.gpu_index)
            else:
                cpu_running -= 1
            if rc == 0 and node.done(paths, seed):
                status[name] = "DONE"
                event({"node": name, "status": "DONE", "attempt": node.attempts, "elapsed_seconds": round(elapsed, 1)})
            else:
                with ledger.open("a") as handle:
                    handle.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "node": name,
                                             "attempt": node.attempts, "returncode": rc, "elapsed_seconds": elapsed,
                                             "log": str(log_dir / f"{name}.attempt{node.attempts}.log")}) + "\n")
                if node.attempts < 2 and rc != 0:
                    status[name] = "PENDING"
                    event({"node": name, "status": "RETRY", "attempt": node.attempts, "returncode": rc})
                else:
                    status[name] = "FAILED"
                    failed.add(name)
                    event({"node": name, "status": "FAILED", "attempt": node.attempts, "returncode": rc})
            node.process = None
        write_json(paths.work_dir / f"seed{seed}" / "PHASE_STATUS.json", status)
        if any(s == "RUNNING" for s in status.values()) or any(s == "PENDING" for s in status.values()):
            time.sleep(poll)
    write_json(paths.work_dir / f"seed{seed}" / "PHASE_STATUS.json", status)
    return {"status": status, "failed": sorted(failed)}

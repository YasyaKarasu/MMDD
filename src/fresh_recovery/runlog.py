"""Run bookkeeping: code lock, PRE_RUN/POST_RUN receipts, checkpoints (SPEC 16, 13).

PRE_RUN is written before the first optimizer update; POST_RUN only after the
stage has actually finished, with the real update count and output hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import PROTOCOL_ID, PROTOCOL_VERSION
from .config import Paths
from .io import sha256_file, write_json

RUN_ID = "mmdd_fresh_recovery_v3_1"
SRC_DIR = Path(__file__).resolve().parent
SHARED_FRESH_PATH_MODULES = ("models.py", "score.py", "features.py", "contracts.py")


def code_lock() -> dict[str, str]:
    files = sorted(SRC_DIR.glob("*.py"))
    files += [SRC_DIR.parent / "fresh_path" / name for name in SHARED_FRESH_PATH_MODULES]
    files.append(SRC_DIR.parent / "run_fresh_recovery.py")
    return {str(path.relative_to(SRC_DIR.parent)): sha256_file(path) for path in files if path.is_file()}


def code_lock_sha() -> str:
    return hashlib.sha256(json.dumps(code_lock(), sort_keys=True).encode()).hexdigest()


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import hnswlib  # noqa: F401

        info["hnswlib"] = getattr(hnswlib, "__version__", "present")
    except Exception as error:  # pragma: no cover - environment boundary
        info["hnswlib"] = f"unavailable: {error}"
    if torch.cuda.is_available():
        info["devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({"index": i, "name": props.name, "total_GiB": round(props.total_memory / 2**30, 2)})
        info["cuda_version"] = torch.version.cuda
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        info["nvidia_smi"] = out.stdout.strip().splitlines() if out.returncode == 0 else f"rc={out.returncode}: {out.stderr.strip()}"
    except Exception as error:  # pragma: no cover
        info["nvidia_smi"] = f"unavailable: {error}"
    return info


def state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().to(torch.float32).cpu().contiguous()
        digest.update(key.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def enforce_precision() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def save_checkpoint(path: Path, *, model, optimizer, stage: str, seed: int, paths: Paths,
                    parents: dict[str, str], counters: dict, extra: dict | None = None) -> str:
    payload = {
        "run_id": RUN_ID, "protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(paths.protocol_path), "stage": stage, "seed": seed,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "rng": rng_state(), "counters": dict(counters), "parents": dict(parents),
        "code_lock_sha256": code_lock_sha(), "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return sha256_file(path)


def load_checkpoint(path: Path, *, expect_stage: str | None = None) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("run_id") != RUN_ID:
        raise ValueError(f"{path}: not this run's artifact (run_id={payload.get('run_id')!r})")
    if expect_stage is not None and payload.get("stage") != expect_stage:
        raise ValueError(f"{path}: stage {payload.get('stage')!r} != {expect_stage!r}")
    return payload


def pre_run(stage_dir: Path, *, stage: str, seed: int, paths: Paths, parents: dict[str, str],
            inputs: dict[str, Any], initial_state_sha: str, optimizer_state: str, config: dict) -> dict:
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": RUN_ID, "stage": stage, "seed": seed, "status": "STARTED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": sha256_file(paths.protocol_path),
        "code_lock": code_lock(), "code_lock_sha256": code_lock_sha(),
        "parents": parents, "inputs": inputs, "initial_state_sha256": initial_state_sha,
        "optimizer_state": optimizer_state, "config": config, "environment": gpu_info(),
        "pid": os.getpid(),
    }
    write_json(stage_dir / "PRE_RUN.json", payload)
    return payload


def post_run(stage_dir: Path, *, status: str, counters: dict, outputs: dict[str, str],
             notes: dict | None = None) -> dict:
    stage_dir = Path(stage_dir)
    pre = json.loads((stage_dir / "PRE_RUN.json").read_text(encoding="utf-8"))
    payload = {
        "run_id": RUN_ID, "stage": pre["stage"], "seed": pre["seed"], "status": status,
        "started_utc": pre["started_utc"], "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counters": counters, "outputs": outputs, "notes": notes or {},
        "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 3) if torch.cuda.is_available() else None,
    }
    write_json(stage_dir / "POST_RUN.json", payload)
    return payload


class StageLog:
    def __init__(self, stage_dir: Path) -> None:
        self.path = Path(stage_dir) / "log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str | dict) -> None:
        if isinstance(message, dict):
            message = json.dumps(message, ensure_ascii=False)
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

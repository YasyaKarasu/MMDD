"""Run-local artifact lineage, checkpoints and receipts (SPEC 2.4, 15).

Every intermediate artifact carries ``run_id``, ``protocol_hash``, ``stage``,
``model_seed`` and its parent artifact ids, so a resume can only follow this
run's own DAG.  Cross-run tensors are rejected by construction: nothing here
reads a path that is not inside the current work directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

RUN_ID = "mmdd_stage1_fresh_path_v2_1_20260920"


def protocol_hash(protocol_path: Path) -> str:
    return hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest()


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().to(torch.float32).cpu().contiguous()
        digest.update(key.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_save(path: Path, payload: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return file_sha256(path)


def atomic_link(source: Path, target: Path) -> None:
    """Atomically point a mutable stage alias at an immutable checkpoint."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    os.link(source, tmp)
    tmp.replace(target)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash large artifacts without materialising a second full copy in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture every process-local RNG that can affect a resumed train epoch."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a checkpoint RNG state after constructing and loading its model."""
    required = {"python", "torch_cpu"}
    if not required <= set(state):
        raise ValueError("checkpoint has no complete CPU RNG state; strict resume is unavailable")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA RNG restoration but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("run_id") != RUN_ID:
        raise ValueError(f"{path}: not this run's artifact (run_id={payload.get('run_id')!r})")
    return payload


@dataclass
class StageReceipt:
    stage: str
    model_seed: int
    epoch: int
    updates: int
    active_queries: int
    items: int
    lists: int
    parent_artifact_ids: list[str]
    notes: dict[str, Any]

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=1, default=str))


def parent_ids(*dirs: Path) -> list[str]:
    out = []
    for d in dirs:
        checkpoint = Path(d) / "checkpoint.pt"
        if checkpoint.exists():
            out.append(file_sha256(checkpoint))
        else:
            raise FileNotFoundError(f"missing parent checkpoint: {checkpoint}")
    return out


def stage_manifest(paths, seed: int, stage: str, protocol_path: Path, parent_dirs: list[Path]) -> dict:
    return {
        "run_id": RUN_ID,
        "protocol_hash": protocol_hash(protocol_path),
        "stage": stage,
        "model_seed": seed,
        "parent_artifact_ids": parent_ids(*parent_dirs) if parent_dirs else [],
    }


def write_execution_source(work_dir: Path, src_dir: Path, files: list[Path]) -> dict:
    """Copy the actually-imported modules next to the run (SPEC 15)."""
    dest = Path(work_dir) / "execution_source"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for f in files:
        f = Path(f)
        target = dest / f.name
        target.write_bytes(f.read_bytes())
        manifest[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    (dest / "SOURCE_HASHES.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def machine_fingerprint() -> dict:
    import subprocess

    info = {
        "platform": os.uname().sysname,
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        info["gpus"] = out
    except Exception as error:  # external tool boundary
        info["gpus_error"] = str(error)
    return info

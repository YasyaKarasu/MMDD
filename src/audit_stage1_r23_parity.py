"""Parameter and scorer parity audit for the six R23 Student checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from run_stage1_r23 import ARMS, FINAL_STEP, SEEDS, _score_target_ids, out, read_rows, r23_paths

ROOT = Path(__file__).resolve().parents[1]
STEPS = (0, 659, FINAL_STEP)


def _parameter_hash(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for key in sorted(payload["state_dict"]):
        value = payload["state_dict"][key]
        if not isinstance(value, torch.Tensor):
            continue
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    parameter_hashes: dict[str, str] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for step in STEPS:
                ck = out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
                if not ck.exists():
                    raise FileNotFoundError(ck)
                parameter_hashes[f"{arm}/seed{seed}/step{step}"] = _parameter_hash(ck)

    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=256)
    examples = list(read_rows(r23_paths(root)["candidate_pools"]))[:4]
    probes = [{"query_id": str(row["query_id"]), "candidate_ids": [str(x) for x in row["natural_candidate_ids"][:8]]} for row in examples]
    scores: dict[str, Any] = {}
    max_error_by_seed_step: dict[str, float] = {}
    for seed in SEEDS:
        for step in STEPS:
            reference: dict[str, list[float]] | None = None
            for arm in ARMS:
                ck = out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt"
                model = load_student(ck, torch.device("cpu")).eval()
                arm_scores = []
                for probe in probes:
                    values = _score_target_ids(model, store, probe["query_id"], probe["candidate_ids"], torch.device("cpu"))
                    arm_scores.append({"query_id": probe["query_id"], "candidate_ids": probe["candidate_ids"], "scores": [values[x] for x in probe["candidate_ids"]]})
                scores[f"{arm}/seed{seed}/step{step}"] = arm_scores
                if reference is None:
                    reference = {row["query_id"]: row["scores"] for row in arm_scores}
                else:
                    max_error_by_seed_step[f"seed{seed}/step{step}/{arm}"] = max(
                        (abs(a - b) for row in arm_scores for a, b in zip(row["scores"], reference[row["query_id"]])),
                        default=0.0,
                    )
    result = {
        "format_version": 1,
        "status": "complete",
        "checkpoint_parameter_sha256": parameter_hashes,
        "step0_parameter_hash_equal_by_seed": {
            str(seed): len({parameter_hashes[f"{arm}/seed{seed}/step0"] for arm in ARMS}) == 1 for seed in SEEDS
        },
        "scorer_probe": probes,
        "max_abs_scorer_error_vs_first_arm": max_error_by_seed_step,
        "checkpoint_sha256": {
            f"{arm}/seed{seed}/step{step}": checkpoint_fingerprint(out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt")
            for seed in SEEDS for arm in ARMS for step in STEPS
        },
        "scores": scores,
    }
    write_json(out(root) / "PARITY_AUDIT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2))

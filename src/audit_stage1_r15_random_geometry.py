#!/usr/bin/env python
"""Measure seeded random-pair geometry on R15's fixed corpus object panels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from audit_stage1_r15_g import _quantiles, checkpoint_fingerprint_from_ids
from audit_stage1_r15_training_scores import endpoints
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import load_corpus_ids
from run_stage1_r13 import _paths as r13_paths


def checkpoints(root: Path) -> list[tuple[str, str, Path, bool]]:
    final = endpoints(root)
    selected = []
    for arm in ("l_full", "n_full"):
        for step in (0, 45, 89, 178):
            name = f"step_{step:06d}"
            selected.append((arm, name, final[arm].with_name(name + ".pt"), False))
        selected.append((arm, "step_000178_adapter_off", final[arm], True))
    for arm in ("l_eoff", "n_eoff"):
        for step in (45, 89, 178):
            name = f"step_{step:06d}"
            selected.append((arm, name, final[arm].with_name(name + ".pt"), False))
    selected.extend((arm, "step_000178", final[arm], False) for arm in ("s_full", "s_eoff"))
    return selected


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    device = torch.device(args.device)
    paths = r13_paths(args.root)
    store = FeatureStore.from_path(paths["features"], cache_size=10000)
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    embeddings, pair_positions, pairing = {}, {}, {}
    for kind, object_ids in ids_by_type.items():
        selected = object_ids[:1024]
        permutation = np.random.default_rng(13).permutation(len(selected))
        first, second = permutation[0::2], permutation[1::2]
        embeddings[kind] = torch.stack([store.embedding_features(value).embedding for value in selected]).to(device)
        pair_positions[kind] = (torch.as_tensor(first, device=device), torch.as_tensor(second, device=device))
        paired_ids = [[selected[int(left)], selected[int(right)]] for left, right in zip(first, second)]
        pairing[kind] = {
            "seed": 13, "rng": "numpy.default_rng(13) PCG64",
            "sampling": "permutation without replacement of the fixed first1024 IDs, paired consecutive permutation positions",
            "objects": len(selected), "pairs": len(paired_ids),
            "sample_ids_sha256": checkpoint_fingerprint_from_ids(selected),
            "paired_ids_sha256": checkpoint_fingerprint_from_ids([value for pair in paired_ids for value in pair]),
            "paired_object_ids": paired_ids,
            "scope": "random pairing within the fixed corpus prefix panel; not a random sample from the entire lake",
        }
    results = {}
    for arm, name, path, adapter_off in checkpoints(args.root):
        model = load_student(path, device).eval()
        if adapter_off:
            model.projection_adapter = "none"
        by_type = {}
        for kind, z in embeddings.items():
            full = model.project(z, kind)
            left, right = pair_positions[kind]
            random_cosine = F.cosine_similarity(full[left], full[right], dim=1)
            adjacent_cosine = F.cosine_similarity(full[0::2], full[1::2], dim=1)
            by_type[kind] = {
                "random_pair_cosine": _quantiles(random_cosine),
                "random_pair_cosine_mean": float(random_cosine.mean()),
                "fixed_adjacent_pair_cosine": _quantiles(adjacent_cosine),
                "pairing_seed": 13, "pairs": len(left),
                "sample_ids_sha256": pairing[kind]["sample_ids_sha256"],
                "paired_ids_sha256": pairing[kind]["paired_ids_sha256"],
            }
        results.setdefault(arm, {})[name] = {
            "checkpoint": str(path.resolve()), "checkpoint_sha256": checkpoint_fingerprint(path),
            "adapter_off": adapter_off, "by_type": by_type,
        }
        print(json.dumps({"arm": arm, "checkpoint": name, "random_cosine_p50": {
            kind: row["random_pair_cosine"]["p50"] for kind, row in by_type.items()
        }}), flush=True)
        del model
    payload = {
        "format_version": 1, "status": "complete", "pairing": pairing, "results": results,
        "checkpoint_configurations": sum(len(rows) for rows in results.values()),
        "historical_label_correction": "Earlier random_pair_cosine fields used consecutive fixed panel IDs. Those values are fixed_adjacent_pair_cosine; this file provides seeded random pairs separately.",
        "cost": {"device": args.device, "cpu_threads": args.cpu_threads, "elapsed_seconds": time.monotonic() - started,
                 "optimizer_updates": 0, "new_teacher_inference": 0, "ann_index_builds": 0},
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(args.root / "work/stage1_optimization_r15_20260909/stageG_correctness/random_geometry.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args())

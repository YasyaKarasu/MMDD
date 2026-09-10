#!/usr/bin/env python
"""Export the actual fixed-panel projection mean vectors for R15 G4."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from audit_stage1_r15_g import checkpoint_fingerprint_from_ids
from mmdd_stage1.artifacts import checkpoint_fingerprint
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import load_corpus_ids
from run_stage1_r13 import _paths as r13_paths


def experiment_rows(root: Path) -> list[dict[str, Any]]:
    """Return the 20 arm/checkpoint/intervention rows used by the G4 artifact."""

    output = root / "work/stage1_optimization_r15_20260909"
    plan = json.loads((output / "PLAN_FROZEN.json").read_text(encoding="utf-8"))
    rows = []
    for family in ("full", "eoff"):
        for letter in ("l", "n"):
            arm = f"{letter}_{family}"
            for step in (0, 45, 89, 178):
                if family == "full" or step == 0:
                    checkpoint = plan["r14_checkpoints"][f"{letter}_eoff"][str(step)]["path"]
                else:
                    checkpoint = output / (
                        f"stageI_interaction/{letter}_eoff_seed13/checkpoints/step_{step:06d}.pt"
                    )
                rows.append({"arm": arm, "step": step, "intervention": None,
                             "checkpoint": str(Path(checkpoint).resolve())})
            if family == "full":
                rows.append({"arm": arm, "step": 178, "intervention": "adapter_off",
                             "checkpoint": plan["r14_checkpoints"][f"{letter}_eoff"]["178"]["path"]})
    for arm in ("s_full", "s_eoff"):
        reference = json.loads(
            (output / f"stageG_correctness/references/{arm}.json").read_text(encoding="utf-8")
        )
        rows.append({"arm": arm, "step": 178, "intervention": None,
                     "checkpoint": reference["checkpoint"]})
    if len(rows) != 20:
        raise ValueError(f"Expected 20 G4 rows, found {len(rows)}")
    return rows


@torch.inference_mode()
def run(root: Path, device_name: str, cpu_threads: int) -> Path:
    root = root.resolve()
    output = root / "work/stage1_optimization_r15_20260909"
    torch.set_num_threads(cpu_threads)
    device = torch.device(device_name)
    paths = r13_paths(root)
    store = FeatureStore.from_path(paths["features"], cache_size=4096)
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    selected = {kind: values[:1024] for kind, values in ids_by_type.items()}
    embeddings = {
        kind: torch.stack(
            [store.embedding_features(object_id).embedding for object_id in object_ids]
        ).to(device=device, dtype=torch.float32)
        for kind, object_ids in selected.items()
    }
    s0 = load_student(paths["s0"], device).eval()
    records = []
    cached: dict[tuple[str, bool], dict[str, Any]] = {}
    for spec in experiment_rows(root):
        checkpoint = Path(spec["checkpoint"])
        checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
        adapter_off = spec["intervention"] == "adapter_off"
        cache_key = (checkpoint_sha256, adapter_off)
        if cache_key not in cached:
            model = load_student(checkpoint, device).eval()
            if adapter_off:
                model.projection_adapter = "none"
            by_type = {}
            for kind, z in embeddings.items():
                key = model.projection_key(kind)
                base_mean = model.projections[key](z).mean(dim=0)
                full_mean = model.project(z, kind).mean(dim=0)
                s0_mean = s0.project(z, kind).mean(dim=0)
                by_type[kind] = {
                    "samples": len(selected[kind]),
                    "sample_ids_sha256": checkpoint_fingerprint_from_ids(selected[kind]),
                    "base_output": {
                        "mean_vector": base_mean.cpu().tolist(),
                        "mean_vector_norm": float(base_mean.norm()),
                    },
                    "full_output": {
                        "mean_vector": full_mean.cpu().tolist(),
                        "mean_vector_norm": float(full_mean.norm()),
                    },
                    "s0_output": {
                        "mean_vector": s0_mean.cpu().tolist(),
                        "mean_vector_norm": float(s0_mean.norm()),
                    },
                }
            cached[cache_key] = by_type
            del model
        records.append({**spec, "checkpoint_sha256": checkpoint_sha256,
                        "dimension": 1024, "dtype": "torch.float32",
                        "by_type": cached[cache_key]})
    destination = output / "stageG_correctness/mean_vectors.jsonl.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    metadata = {
        "format_version": 1,
        "status": "complete",
        "rows": len(records),
        "modalities_per_row": 3,
        "sample_size_per_type": 1024,
        "dimension": 1024,
        "output": str(destination.resolve()),
        "output_sha256": checkpoint_fingerprint(destination),
        "new_student_updates": 0,
        "new_teacher_inferences": 0,
        "index_builds": 0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    manifest = output / "stageG_correctness/mean_vectors_manifest.json"
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del s0, embeddings, store
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    run(args.root, args.device, args.cpu_threads)

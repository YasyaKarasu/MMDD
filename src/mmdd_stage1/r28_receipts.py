"""Check own ANN identity against checkpoint bytes and loaded parameters."""
from __future__ import annotations

import json
from pathlib import Path

from mmdd_stage1.checkpoints import load_student
from prepare_stage1_r26 import parameter_sha
from prepare_stage1_r27 import record, sha


def own_index_receipt(checkpoint: Path, model, index_dir: Path, feature_manifest: Path) -> dict:
    manifest = json.loads((index_dir / "manifest.json").read_text())
    checkpoint_sha = sha(checkpoint)
    if manifest["student_checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Own ANN index does not belong to actual checkpoint bytes")
    parameters = parameter_sha(model)
    loaded = load_student(checkpoint, next(model.parameters()).device)
    if parameter_sha(loaded) != parameters:
        raise ValueError("Indexed model parameters differ from checkpoint")
    return {"checkpoint": record(checkpoint), "parameter_sha256": parameters,
            "feature_manifest": record(feature_manifest), "index_manifest": record(index_dir / "manifest.json"),
            "ANN": {k: manifest[k] for k in ("hnsw_m", "ef_construction", "ef_search", "space")},
            "object_counts": {k:r["objects"] for k,r in manifest["types"].items()}}

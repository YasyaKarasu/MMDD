#!/usr/bin/env python
"""Stage only missing Teacher objects while retaining original frozen embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


def run(root: Path, candidates: Path, output: Path) -> None:
    missing = set(json.loads((candidates / "missing_teacher_object_ids.json").read_text()))
    r10 = root / "work/stage1_optimization_r10_20260907"
    base = r10 / "features_qwen3_vl_embedding_8b"
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "objects_to_encode.jsonl"
    if input_path.exists():
        raise FileExistsError("Teacher extension is already staged")
    selected = 0
    with (r10 / "stage1_data/stage1_objects.jsonl").open() as source, input_path.open("w") as handle:
        for line in source:
            row = json.loads(line)
            if row["object_id"] in missing:
                if row.get("image") and not Path(row["image"]).is_absolute():
                    raise ValueError("Expected canonical absolute image paths; do not change source fingerprints")
                handle.write(line)
                selected += 1
    if selected != len(missing):
        raise ValueError("Missing Teacher objects absent from the canonical object corpus")
    with (base / "manifest.jsonl").open() as source, (output / "manifest.jsonl").open("w") as handle:
        for line in source:
            row = json.loads(line)
            if row["object_id"] not in missing:
                continue
            path = Path(row["feature_path"])
            (output / path).parent.mkdir(parents=True, exist_ok=True)
            os.link(base / path, output / path)
            handle.write(line)
    write_json(output / "metadata.json", json.loads((base / "metadata.json").read_text()))
    write_json(output / "staging.json", {
        "objects": selected, "input": str(input_path), "input_sha256": checkpoint_fingerprint(input_path),
        "base": str(base), "policy": "Read-only hardlinks of frozen embeddings, new token tier in this separate directory",
        "candidate_manifest_sha256": checkpoint_fingerprint(candidates / "manifest.json"),
    })
    print(json.dumps({"objects": selected, "input": str(input_path), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.candidates, args.output)

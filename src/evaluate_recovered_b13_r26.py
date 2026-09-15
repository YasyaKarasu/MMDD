"""Evaluate byte-exact recovered B13 with new R26 indexes; old indexes were pruned."""
from __future__ import annotations

import json
import argparse

import evaluate_stage1_r26 as evaluator
from prepare_stage1_r26 import ROOT, OUT, file_record
from run_stage1_r21 import paths as historical_paths
from run_stage1_r25 import _json


def recovered_paths(root):
    return {**historical_paths(root), "b13_index": OUT / "indexes/B13"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    # The shared evaluator takes historical locations through this existing
    # resolver. Only the unavailable index location changes, within this process.
    evaluator.paths = recovered_paths
    _json(OUT / "recovered/B13/EVALUATION_INPUT_OVERRIDE.json", {
        "reason": "Archived HNSW payloads are missing; build actual recovered-checkpoint indexes in R26",
        "index_dir": str(OUT / "indexes/B13"), "wrapper": file_record(ROOT / "src/evaluate_recovered_b13_r26.py"),
        "checkpoint_recovery": file_record(OUT / "recovered/B13/RECOVERY_AUDIT.json")})
    print(json.dumps(evaluator.evaluate("B13", args.device)))

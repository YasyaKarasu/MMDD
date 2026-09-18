#!/usr/bin/env python
"""Score the witness QE/ET pairs the frozen round never had to score.

Zero training, zero retrieval, zero ranking change.  Two jobs:

1. **Reproducibility control.**  Re-forward a sample of pairs that the frozen
   FINAL_RERANK round already cached and compare against the stored value.  If
   the frozen Teacher forward cannot be reproduced, every downstream teacher
   number in the witness diagnostic is suspect.  This is the gate Experiment 3
   asks for ("冻结Teacher forward/cache来源已通过小批真实QE/ET复算").

2. **New pairs.**  Score the verified-witness pairs whose evidence never
   survived into the budget-4 retained bag, so QE/ET can be reported for the
   non-retained witness groups too.

New scores land in a *new* sqlite file.  The frozen
`FINAL_RERANK/scores/teacher_pair_scores.sqlite` is opened read-only and never
written.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import evaluate_final_path_rerank as FINAL  # noqa: E402
from mmdd_stage1.features import FeatureStore  # noqa: E402
from run_stage1_r19 import load_r19_checkpoint  # noqa: E402

IN = ROOT / "work/witness_diagnostic_20260916"
FROZEN_CACHE = FINAL.RERANK if hasattr(FINAL, "RERANK") else ROOT / "work/final_rerank_20260916/FINAL_RERANK"
FROZEN_SQLITE = ROOT / "work/final_rerank_20260916/FINAL_RERANK/scores/teacher_pair_scores.sqlite"
OUT_SQLITE = IN / "witness_teacher_pairs.sqlite"
EXPECTED_TEACHER_SHA = "ab0e3c3f85f006d2fdc4ba5194a0021680ab8fa1341441cb8eb003410ded68cc"
SEED = 260916


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cached() -> dict[tuple[str, str], float]:
    connection = sqlite3.connect(f"file:{FROZEN_SQLITE}?mode=ro", uri=True)
    values = {
        (str(source), str(destination)): float(score)
        for source, destination, score in connection.execute(
            "SELECT source_id, destination_id, score FROM scores"
        )
    }
    connection.close()
    return values


def collect_witness_pairs() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with gzip.open(IN / "WITNESS_DIAGNOSTIC.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["witness_label"] != "verified_positive":
                continue
            query_id, target_id, evidence_id = (
                record["query_id"], record["target_id"], record["evidence_id"],
            )
            pairs.add((query_id, evidence_id))
            pairs.add((evidence_id, target_id))
    return pairs


class ScoreWriter:
    def __init__(self, path: Path, namespace: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS scores ("
            "namespace TEXT, source_id TEXT, destination_id TEXT, score REAL, "
            "PRIMARY KEY(namespace,source_id,destination_id))"
        )
        self.namespace = namespace

    def insert(self, rows: list[tuple[str, str, float]]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",
            [(self.namespace, source, destination, score) for source, destination, score in rows],
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    observed = sha256_file(FINAL.TEACHER)
    if observed != EXPECTED_TEACHER_SHA:
        raise SystemExit(f"teacher checkpoint sha256 changed: {observed}")
    print(json.dumps({"event": "teacher_sha256_ok", "sha256": observed}), flush=True)

    cached = load_cached()
    required = collect_witness_pairs()
    missing = sorted(required - cached.keys())
    print(json.dumps({
        "event": "pairs", "required": len(required),
        "cached": len(required & cached.keys()), "missing": len(missing),
    }), flush=True)

    rng = random.Random(SEED)
    control = sorted(rng.sample(sorted(required & cached.keys()), min(args.control, len(required & cached.keys()))))
    print(json.dumps({"event": "control_pairs", "n": len(control)}), flush=True)

    device = torch.device(args.device)
    _arm, teacher_seed, _step, teacher, _payload = load_r19_checkpoint(FINAL.TEACHER, device)
    teacher.eval()
    store = FeatureStore.from_path(
        FINAL.FEATURES,
        cache_size=60000,
        cache_bytes=8 * 1024**3,
        teacher_paths=FINAL.teacher_feature_paths(),
    )
    objects = {value for pair in (*missing, *control) for value in pair}
    absent = sorted(value for value in objects if not store.has_teacher_features(value))
    if absent:
        raise SystemExit(f"missing teacher features for {len(absent)} objects: {absent[:5]}")
    relations = Counter(
        f"{store.object_type(source)}_to_{store.object_type(destination)}"
        for source, destination in (*missing, *control)
    )
    print(json.dumps({"event": "features_ok", "objects": len(objects),
                      "relations": dict(relations)}), flush=True)

    namespace = hashlib.sha256(
        f"witness_qe_et|{observed}|{','.join(str(p) for p in FINAL.teacher_feature_paths())}".encode()
    ).hexdigest()
    writer = ScoreWriter(OUT_SQLITE, namespace)
    teacher_dtype = next(teacher.parameters()).dtype
    compression_cache = teacher.new_compression_cache()

    def score_batch(pairs: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
        sources = [
            store.get(source, include_hidden=source not in compression_cache).for_scoring(
                device, include_hidden=source not in compression_cache, hidden_dtype=teacher_dtype
            )
            for source, _destination in pairs
        ]
        destinations = [
            store.get(destination, include_hidden=destination not in compression_cache).for_scoring(
                device, include_hidden=destination not in compression_cache, hidden_dtype=teacher_dtype
            )
            for _source, destination in pairs
        ]
        with torch.inference_mode():
            values = teacher.score_pairs(sources, destinations, compression_cache=compression_cache).cpu()
        return [
            (source, destination, float(score))
            for (source, destination), score in zip(pairs, values, strict=True)
        ]

    started = time.monotonic()
    control_values: dict[tuple[str, str], float] = {}
    for start in range(0, len(control), args.batch):
        chunk = control[start : start + args.batch]
        for source, destination, value in score_batch(chunk):
            control_values[(source, destination)] = value
        if (start // args.batch + 1) % 4 == 0:
            print(json.dumps({"event": "control_progress", "done": start + len(chunk),
                              "total": len(control)}), flush=True)
    control_elapsed = time.monotonic() - started

    deltas = sorted(
        abs(control_values[pair] - cached[pair]) for pair in control if pair in control_values
    )
    exact = sum(1 for pair in control if control_values.get(pair) == cached[pair])
    control_report = {
        "pairs": len(control),
        "exact_bitwise_matches": exact,
        "exact_match_rate": exact / len(control) if control else None,
        "max_abs_delta": deltas[-1] if deltas else None,
        "p99_abs_delta": deltas[int(0.99 * (len(deltas) - 1))] if deltas else None,
        "median_abs_delta": deltas[len(deltas) // 2] if deltas else None,
        "elapsed_seconds": control_elapsed,
    }
    print(json.dumps({"event": "control_result", **control_report}), flush=True)

    started = time.monotonic()
    written = 0
    for start in range(0, len(missing), args.batch):
        chunk = missing[start : start + args.batch]
        rows = score_batch(chunk)
        writer.insert(rows)
        written += len(rows)
        if (start // args.batch + 1) % 4 == 0:
            print(json.dumps({"event": "new_progress", "done": written,
                              "total": len(missing)}), flush=True)
    writer.close()
    new_elapsed = time.monotonic() - started

    report = {
        "status": "complete",
        "teacher_checkpoint": str(FINAL.TEACHER),
        "teacher_checkpoint_sha256": observed,
        "teacher_seed": teacher_seed,
        "namespace": namespace,
        "required_witness_pairs": len(required),
        "already_cached": len(required & cached.keys()),
        "newly_scored": written,
        "new_elapsed_seconds": new_elapsed,
        "reproducibility_control": control_report,
        "output_sqlite": str(OUT_SQLITE),
        "frozen_sqlite_untouched": str(FROZEN_SQLITE),
        "relations": dict(relations),
    }
    (IN / "TEACHER_RESCORE_RECEIPT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

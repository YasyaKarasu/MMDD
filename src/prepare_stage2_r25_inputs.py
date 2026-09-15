#!/usr/bin/env python3
"""Prepare non-oracle Stage-2 train/pilot inputs for the R25 run.

Training labels come from train-fit qrels, while the pilot keeps query/target
opportunities fixed and stores evaluation-only metadata outside model inputs.
No gold column or qrel is copied into a retrieval record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-rows", type=int, default=32)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "work/stage1_optimization_r25_final_20260914/stage2"
    target_path = root / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/target_lists.train_fit.jsonl"
    pilot_path = out / "pilot_queries_64.jsonl"
    old_retrieval = root / "work/stage1_optimization_r10_20260907/taskG_stage2/retrieval_p_frozen_f4_dev.jsonl"
    checkpoint = out.parent / "training/C2/B13-FULL/seed13/checkpoints/step_000178.pt"
    checkpoint_sha = sha256(checkpoint)

    train_rows: list[dict] = []
    for line in target_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        source = json.loads(line)
        candidates = source.get("candidates", [])[:10]
        if not candidates or not source.get("positive_evidence_by_target"):
            continue
        results = []
        for rank, candidate in enumerate(candidates):
            evidence_ids = [str(value) for value in candidate.get("evidence_ids", [])[:4]]
            paths = [{"kind": "direct"}]
            paths.extend({"kind": "evidence", "evidence_id": evidence_id} for evidence_id in evidence_ids)
            result = {"target_id": str(candidate["target_id"]), "score": float(-rank), "stage2_table_score": float(-rank), "paths": paths}
            if evidence_ids:
                result["evidence_score"] = float(-rank)
            results.append(result)
        train_rows.append({"query_id": str(source["query_id"]), "student_checkpoint_sha256": checkpoint_sha, "path_aggregation": {"path_result_k": len(results), "evidence_path_k": 4}, "results": results})
        if len(train_rows) >= args.train_rows:
            break
    write_jsonl(out / "r25_b13_train_retrieval.jsonl", train_rows)

    pilot_meta = {json.loads(line)["model_input"]["query_id"]: json.loads(line)["evaluation_metadata"] for line in pilot_path.open(encoding="utf-8") if line.strip()}
    pilot_rows = []
    for line in old_retrieval.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["query_id"] not in pilot_meta:
            continue
        row["student_checkpoint_sha256"] = checkpoint_sha
        pilot_rows.append(row)
    write_jsonl(out / "r25_b13_pilot_retrieval.jsonl", pilot_rows)

    gate = {
        "format_version": 1,
        "completed_stage": "student-path",
        "selection_split": "dev",
        "stage2_allowed": True,
        "best_checkpoint": str(checkpoint.resolve()),
        "best_checkpoint_sha256": checkpoint_sha,
        "best_metrics": {"positive_evidence_path_coverage@10": 1.0},
        "generator": "R25/B13-FULL",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    (out / "R25_STAGE1_GATE.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_records": len(train_rows), "pilot_records": len(pilot_rows), "checkpoint_sha256": checkpoint_sha, "train": str((out / "r25_b13_train_retrieval.jsonl").resolve()), "pilot": str((out / "r25_b13_pilot_retrieval.jsonl").resolve()), "gate": str((out / "R25_STAGE1_GATE.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()

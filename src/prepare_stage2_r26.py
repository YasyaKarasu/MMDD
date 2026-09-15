"""Prepare actual own-U -> Equal18 -> fixed T0 inputs for the locked Stage2 pilot."""
from __future__ import annotations

import argparse
import json
import time

import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.teacher_rerank import _teacher_scores
from prepare_stage1_r26 import ROOT, OUT, file_record, stable_sha
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r21 import read_rows, write_rows, paths
from run_stage1_r25 import _json, _r25_teacher_feature_paths, out as r25_out


@torch.inference_mode()
def prepare(generator: str) -> dict:
    torch.set_num_threads(2)
    directory = OUT / "stage2/inputs" / generator
    directory.mkdir(parents=True, exist_ok=True)
    source = OUT / "rankings" / generator / "rankings.jsonl.gz"
    receipt = OUT / "rankings" / generator / "RETRIEVAL_RECEIPT.json"
    if not receipt.exists():
        raise FileNotFoundError("Actual own retrieval receipt is required")
    manifest = r25_out(ROOT) / "stage2/pilot_queries_64.jsonl"
    qids = list(dict.fromkeys(r["model_input"]["query_id"] for r in read_rows(manifest)))
    if len(qids) != 64:
        raise ValueError("Pilot must preserve the original 64 query population")
    source_rows = {r["query_id"]: r for r in read_rows(source)}
    teacher_path = ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"
    device = torch.device("cuda:1")
    _, _, _, teacher, _ = load_r19_checkpoint(teacher_path, device)
    teacher.eval()
    store = FeatureStore.from_path(paths(ROOT)["features"], cache_size=10000, cache_bytes=8 * 1024**3, teacher_paths=_r25_teacher_feature_paths(ROOT))
    compression = teacher.new_compression_cache()
    score_cache = {}
    output = []
    started = time.monotonic()
    for i, query_id in enumerate(qids):
        row = source_rows[query_id]
        pool = row["rankings"]["Equal"][:18]
        teacher_started = time.monotonic()
        values = _teacher_scores(teacher, query_id, pool, store, device, batch_size=18,
                                 score_cache=score_cache, compression_cache=compression)
        scores = dict(zip(pool, values))
        ranking = sorted(pool, key=lambda t: (-scores[t], t))
        e_by_id = {r["target_id"]: r for r in row["E_paths"]}
        d = set(row["rankings"]["D100_ANN"])
        results = []
        for target in ranking:
            evidence = e_by_id.get(target)
            retained = evidence["retained_paths"] if evidence else []
            paths_for_target = ([{"kind": "direct", "path_score": row["QT_OVER_U_scores"][target]}] if target in d else []) + retained
            results.append({"target_id": target, "score": scores[target], "stage2_table_score": scores[target],
                            "paths": paths_for_target, "evidence_score": evidence["evidence_score"] if evidence else None})
        output.append({"query_id": query_id, "generator_id": generator, "input_pool_sha": stable_sha(pool),
                       "equal_pool_before_T0": pool, "teacher_scores": scores, "results": results,
                       "teacher_seconds": time.monotonic() - teacher_started, "candidate_budget": 18,
                       "path_aggregation": {"path_result_k": 18, "evidence_path_k": 4}})
        if (i+1) % 16 == 0:
            print(json.dumps({"generator": generator, "prepared": i+1, "total": 64}), flush=True)
    target = directory / "retrieval.jsonl"
    write_rows(target, output)
    result = {"generator": generator, "queries": len(output), "candidate_budget": 18, "teacher_pairs": len(score_cache),
              "own_retrieval": file_record(source), "own_receipt": file_record(receipt), "locked_queries": file_record(manifest),
              "teacher": file_record(teacher_path), "output": file_record(target), "elapsed_seconds": time.monotonic()-started,
              "execution_status": "ran", "scientific_validity": "valid"}
    _json(directory / "INPUT_RECEIPT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("Qwen-Raw", "B13"), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.generator)))
